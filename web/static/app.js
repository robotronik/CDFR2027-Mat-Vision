'use strict';

/**
 * Interface web du mat de vision.
 * Vanilla JS : aucune dépendance, fonctionne hors ligne sur le réseau local.
 */

const POLL_INTERVAL_MS = 600;
const CAL_POLL_INTERVAL_MS = 400;

const $ = (id) => document.getElementById(id);

let tableInfo = null;
let calibrationPolling = null;
let lastObjects = [];

/* ------------------------------------------------------------------ utils */

async function api(path, options = {}) {
  const response = await fetch(path, { cache: 'no-store', ...options });
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).message || detail;
    } catch (_) { /* réponse non JSON */ }
    throw new Error(`${response.status} ${detail}`);
  }
  return response.json();
}

function num(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return Number(value).toFixed(digits);
}

function toast(message, isError = false) {
  const element = $('toast');
  element.textContent = message;
  element.style.color = isError ? 'var(--danger)' : 'var(--accent)';
  element.hidden = false;
  clearTimeout(element._timer);
  element._timer = setTimeout(() => { element.hidden = true; }, 3000);
}

function setConnected(online, detail = '') {
  $('conn-dot').className = 'dot ' + (online ? 'online' : 'offline');
  $('conn-label').textContent = online ? 'API connectée' : 'API injoignable';
  if (!online && detail) $('conn-label').textContent += ` (${detail})`;
}

function className(isOk) {
  return isOk ? 'ok' : 'ko';
}

/* ----------------------------------------------------------------- render */

function renderStatus(status) {
  $('mode-badge').textContent = status.mode || 'idle';
  $('mode-badge').dataset.mode = status.mode || 'idle';

  $('map-title').textContent = 'Plan de la table';
  $('th-x').textContent = 'x (mm)';
  $('th-y').textContent = 'y (mm)';

  const camera = status.camera || {};
  $('st-camera').textContent = camera.opened
    ? `${camera.device} · ${camera.width}×${camera.height}`
    : 'non ouverte';

  const stats = status.stats || {};
  $('st-fps').textContent = `${num(stats.fps)} i/s`;
  $('st-frames').textContent =
    `${stats.frames || 0} (${stats.failed || 0} échecs, ${stats.detections || 0} objets/img)`;
  $('st-uptime').textContent = `${num(status.uptime_s, 0)} s`;

  const table = status.table || {};
  $('st-table').textContent = table.ok
    ? `verrouillé (${(table.used_ids || []).join(', ')})`
    : (table.reason || 'non verrouillé');
  $('st-table').className = className(table.ok);

  $('st-inliers').textContent = `${table.inliers || 0} / ${table.points || 0}`;
  $('st-residual').textContent = table.residual_px === undefined
    ? '—' : `${num(table.residual_px, 2)} px`;

  const pose = table.camera_position;
  if (pose) {
    $('st-pose').textContent =
      `x ${num(pose.x, 0)} · y ${num(pose.y, 0)} · z ${num(pose.z, 0)} · a ${num(pose.a)}°`;
  } else {
    $('st-pose').textContent = '—';
  }

  const intrinsics = status.intrinsics;
  if (intrinsics) {
    const alerte = status.intrinsics_warning ? ' ⚠' : '';
    $('st-intrinsics').textContent =
      `fx ${num(intrinsics.fx, 0)} · rms ${num(intrinsics.rms, 3)} px${alerte}`;
  } else {
    $('st-intrinsics').textContent = 'non calibrée';
  }

  const messages = [status.error, status.intrinsics_warning].filter(Boolean);
  $('st-error').hidden = messages.length === 0;
  $('st-error').textContent = messages.join(' — ');

  $('hud').textContent = [
    `repère ${table.ok ? 'verrouillé' : 'non verrouillé'} · ` +
      `${table.inliers || 0}/${table.points || 0} inliers · résidu ${num(table.residual_px, 2)} px`,
    `objets ${lastObjects.length} · intrinsèques ${intrinsics ? 'ok' : 'non calibrée'}`,
  ].join('\n');

  $('btn-start').disabled = status.mode === 'detect';
  $('btn-stop').disabled = status.mode === 'idle';
}

function renderObjects(list) {
  lastObjects = list;
  const body = $('objects-body');
  const word = 'objet';
  $('objects-count').textContent = `${list.length} ${word}${list.length > 1 ? 's' : ''}`;

  if (!list.length) {
    body.innerHTML = `<tr><td colspan="5" class="empty">Aucun ${word} détecté</td></tr>`;
    return;
  }

  body.innerHTML = list.map((item) => `
    <tr>
      <td>${escapeHtml(item.label)}</td>
      <td><span class="tag-chip">${item.id}</span></td>
      <td>${num(item.x)}</td>
      <td>${num(item.y)}</td>
      <td>${num(item.a)}°</td>
    </tr>`).join('');
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, (char) => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  })[char]);
}

/* -------------------------------------------------------------- plan table */

function drawMap(objects, status) {
  const canvas = $('map');
  if (!tableInfo) return;

  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width || !height) return;
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }

  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);

  // Repère table en millimètres, +y vers le haut de l'image.
  const fieldWidth = tableInfo.width_mm || 2000;
  const fieldHeight = tableInfo.height_mm || 3000;
  const padding = 12;
  const scale = Math.min((width - 2 * padding) / fieldWidth, (height - 2 * padding) / fieldHeight);
  const centreX = width / 2;
  const centreY = height / 2;

  const sx = (x) => centreX + x * scale;
  const sy = (y) => centreY - y * scale;

  // Fond / tapis
  ctx.fillStyle = '#0f1620';
  ctx.strokeStyle = '#2a3240';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.rect(sx(-fieldWidth / 2), sy(fieldHeight / 2), fieldWidth * scale, fieldHeight * scale);
  ctx.fill();
  ctx.stroke();

  // Axes
  drawArrow(ctx, sx(0), sy(0), sx(500), sy(0), '#f85149');
  drawArrow(ctx, sx(0), sy(0), sx(0), sy(500), '#58a6ff');

  // Tags de coin
  ctx.fillStyle = 'rgba(88, 166, 255, 0.85)';
  ctx.font = '9px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  for (const marker of tableInfo.markers || []) {
    const side = Math.max(5, (marker.size || 100) * scale);
    ctx.fillRect(sx(marker.x) - side / 2, sy(marker.y) - side / 2, side, side);
    ctx.fillStyle = 'rgba(139, 151, 168, 0.9)';
    ctx.fillText(String(marker.id), sx(marker.x), sy(marker.y) - side / 2 - 7);
    ctx.fillStyle = 'rgba(88, 166, 255, 0.85)';
  }

  // Tags / objets détectés
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  const locked = !!(status && status.table && status.table.ok);
  for (const item of objects) {
    const px = sx(item.x);
    const py = sy(item.y);

    // trait d'orientation
    const angle = (item.a || 0) * Math.PI / 180;
    ctx.strokeStyle = 'rgba(63, 185, 80, 0.75)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(px + Math.cos(angle) * 22, py - Math.sin(angle) * 22);
    ctx.stroke();

    ctx.fillStyle = '#3fb950';
    ctx.beginPath();
    ctx.arc(px, py, 6.5, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#e6edf3';
    ctx.font = '10px system-ui, sans-serif';
    ctx.fillText(String(item.label ?? item.id), px, py - 14);
  }

  $('map-scale').textContent = locked
    ? `échelle ${(scale * 100).toFixed(2)} px / 100 mm`
    : 'repère non verrouillé';
}

function drawArrow(ctx, fromX, fromY, toX, toY, color) {
  const angle = Math.atan2(toY - fromY, toX - fromX);
  const head = 7;
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  ctx.moveTo(fromX, fromY);
  ctx.lineTo(toX, toY);
  ctx.stroke();
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.moveTo(toX, toY);
  ctx.lineTo(toX - head * Math.cos(angle - 0.4), toY - head * Math.sin(angle - 0.4));
  ctx.lineTo(toX - head * Math.cos(angle + 0.4), toY - head * Math.sin(angle + 0.4));
  ctx.closePath();
  ctx.fill();
}

/* ------------------------------------------------------------- calibration */

function renderCalibration(payload) {
  const target = payload.target || 0;
  const captured = payload.captured || 0;
  const percent = target ? Math.min(100, (captured / target) * 100) : 0;

  $('cal-bar').style.width = `${percent}%`;
  $('cal-status').textContent = payload.status || 'idle';
  $('cal-frames').textContent = `${captured} / ${target}`;
  $('cal-sharpness').textContent = num(payload.sharpness, 0);
  $('cal-found').textContent = payload.found ? 'détecté' : 'non détecté';
  $('cal-message').textContent = payload.message || '—';

  const intrinsics = payload.intrinsics;
  if (intrinsics) {
    $('cal-rms').textContent = `${num(intrinsics.rms, 3)} px`;
    $('cal-fx').textContent = `${num(intrinsics.fx, 0)} / ${num(intrinsics.fy, 0)}`;
  } else {
    $('cal-rms').textContent = '—';
    $('cal-fx').textContent = '—';
  }

  const running = payload.status === 'running';
  $('btn-cal-start').disabled = running;
  $('btn-cal-stop').disabled = !running;

  // Sortie d'état : on arrête le rafraîchissement et on annonce le résultat.
  if (!running && calibrationPolling) {
    stopCalibrationPolling();
    if (payload.status === 'done') {
      toast(`Calibration terminée — ${num(payload.rms, 3)} px d'erreur`);
    } else if (payload.status === 'failed') {
      toast(`Calibration échouée : ${payload.message}`, true);
    }
  }
}

function startCalibrationPolling() {
  stopCalibrationPolling();
  calibrationPolling = setInterval(async () => {
    try {
      renderCalibration(await api('/calibration/status'));
    } catch (error) {
      toast(`Calibration : ${error.message}`, true);
      stopCalibrationPolling();
    }
  }, CAL_POLL_INTERVAL_MS);
}

function stopCalibrationPolling() {
  if (calibrationPolling) {
    clearInterval(calibrationPolling);
    calibrationPolling = null;
  }
}

/* ------------------------------------------------------------------- flux */

const STREAM_MAX_RETRIES = 6;

let streamRetries = 0;

function streamUrl() {
  return `/stream?fps=12&quality=${$('quality-select').value}&t=${Date.now()}`;
}

function setStream(enabled) {
  const image = $('stream');
  const placeholder = $('stream-placeholder');
  $('stream-toggle').checked = enabled;
  if (enabled) {
    streamRetries = 0;
    placeholder.hidden = true;
    image.hidden = false;
    image.src = streamUrl();
  } else {
    image.removeAttribute('src');
    image.hidden = true;
    placeholder.hidden = false;
  }
}

function showStreamPlaceholder(message) {
  $('stream').hidden = true;
  $('stream-placeholder').hidden = false;
  $('stream-placeholder').querySelector('p').textContent = message;
}

/**
 * Le flux MJPEG peut échouer ponctuellement (caméra encore en chauffe au
 * démarrage du serveur, clé USB débranchée…). On retente quelques fois avant
 * d'annoncer que le flux est indisponible.
 */
function onStreamError() {
  if (!$('stream-toggle').checked) return;
  if (streamRetries >= STREAM_MAX_RETRIES) {
    showStreamPlaceholder('Flux indisponible — caméra hors ligne ?');
    return;
  }
  streamRetries += 1;
  setTimeout(() => {
    if ($('stream-toggle').checked) $('stream').src = streamUrl();
  }, 600 * streamRetries);
}

/* -------------------------------------------------------------------- API */

async function command(path, successMessage) {
  try {
    const payload = await api(path, { method: 'POST' });
    toast(successMessage || payload.message || 'OK');
    await refresh();
  } catch (error) {
    toast(error.message, true);
  }
}

/* --------------------------------------------------------------- boucle */

async function refresh() {
  try {
    const [status, objects] = await Promise.all([api('/status'), api('/objects')]);
    const list = objects.objects || [];
    renderObjects(list);
    renderStatus(status);
    drawMap(list, status);
    setConnected(true);
  } catch (error) {
    setConnected(false, error.message);
  }
}

function loop() {
  refresh().finally(() => setTimeout(loop, POLL_INTERVAL_MS));
}

/* -------------------------------------------------------------- démarrage */

function wireUp() {
  $('btn-start').addEventListener('click', () => command('/start', 'Détection démarrée'));
  $('btn-stop').addEventListener('click', () => command('/stop', 'Détection arrêtée'));
  $('btn-reset').addEventListener('click', () => command('/reset', 'Suivi réinitialisé'));
  $('btn-snapshot').addEventListener('click', async () => {
    try {
      const payload = await api('/snapshot', { method: 'POST' });
      toast(`Image enregistrée : ${payload.path}`);
    } catch (error) {
      toast(error.message, true);
    }
  });

  $('btn-shutdown').addEventListener('click', async () => {
    const question =
      'Arrêter complètement le serveur du mat de vision ?\n\n' +
      'La caméra sera libérée et cette interface ne répondra plus.';
    if (!window.confirm(question)) return;

    setStream(false); // ne plus solliciter un serveur qui s'arrête
    try {
      await api('/shutdown', { method: 'POST' });
      toast('Serveur arrêté — relancez « python server.py »');
    } catch (error) {
      toast(error.message, true);
    }
  });

  $('btn-cal-start').addEventListener('click', async () => {
    try {
      await api('/calibration/start', { method: 'POST' });
      toast('Calibration lancée — présentez le damier');
      startCalibrationPolling();
    } catch (error) {
      toast(error.message, true);
    }
  });

  $('btn-cal-stop').addEventListener('click', async () => {
    try {
      await api('/calibration/stop', { method: 'POST' });
      toast('Calibration annulée');
      stopCalibrationPolling();
      renderCalibration(await api('/calibration/status'));
      await refresh();
    } catch (error) {
      toast(error.message, true);
    }
  });

  $('stream-toggle').addEventListener('change', (event) => setStream(event.target.checked));
  $('stream-resume').addEventListener('click', () => setStream(true));
  $('quality-select').addEventListener('change', () => {
    if ($('stream-toggle').checked) setStream(true);
  });

  $('stream').addEventListener('error', onStreamError);

  window.addEventListener('resize', () => drawMap([], null));
}

async function init() {
  wireUp();
  setStream(true);
  try {
    tableInfo = await api('/table');
  } catch (_) {
    tableInfo = { width_mm: 2000, height_mm: 3000, markers: [] };
  }
  try {
    renderCalibration(await api('/calibration/status'));
    const status = await api('/calibration/status');
    if (status.status === 'running') startCalibrationPolling();
    $('cal-config').textContent =
      `${status.config.cols}×${status.config.rows} coins · case ${status.config.square_size_mm} mm · ` +
      `${status.config.target_frames} vues`;
  } catch (_) { /* calibration indisponible */ }

  loop();
}

// Le canvas de la carte ne peut être redimensionné que lorsque son onglet est
// visible : on rafraîchit donc l'affichage au retour sur l'onglet Vision.
document.addEventListener('tabchange', (event) => {
  if (event.detail.tab === 'vision') refresh();
});

document.addEventListener('DOMContentLoaded', init);
