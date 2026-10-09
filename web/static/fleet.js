'use strict';

/**
 * Onglets « Stratégie » et « Live table ».
 *
 * Stratégie : envoie couleur et stratégie aux robots (principal, chasseur,
 * essaim) via l'API du mat, qui relaie vers chaque robot, et se resynchronise
 * régulièrement pour refléter un changement fait directement sur le robot.
 *
 * Live table : dessine sur un canvas le plan de la table (map de l'année en
 * fond), les robots avec leur trajectoire, le robot adverse et les objets de
 * jeu détectés par le mat.
 */
(() => {
  const $ = (id) => document.getElementById(id);

  const GROUPS = ['main', 'hunter', 'swarm'];
  const GROUP_TITLES = { main: 'Robot principal', hunter: 'Robot chasseur', swarm: 'Essaim' };
  const ROBOT_COLORS = { blue: '#1f6feb', yellow: '#e3b341', '': '#8b97a8' };
  const STRATEGY_POLL_MS = 3000;
  const LIVE_POLL_MS = 300;
  const MAP_DEFAULT = '/static/map.svg';

  let currentTab = 'strategy';
  let fleetData = null;
  let strategySignature = '';
  let livePayload = null;
  let mapImage = null;
  let mapSource = '';
  let strategyTimer = null;
  let liveTimer = null;

  /* ----------------------------------------------------------------- utils */

  async function api(path, options = {}) {
    const response = await fetch(path, { cache: 'no-store', ...options });
    if (!response.ok) {
      let detail = response.statusText;
      try { detail = (await response.json()).message || detail; } catch (_) { /* non JSON */ }
      throw new Error(detail);
    }
    return response.json();
  }

  function post(path, payload) {
    return api(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  }

  function fleetToast(message, isError = false) {
    const element = $('fleet-toast');
    if (!element) return;
    element.textContent = message;
    element.style.color = isError ? 'var(--danger)' : 'var(--accent)';
    element.hidden = false;
    clearTimeout(element._timer);
    element._timer = setTimeout(() => { element.hidden = true; }, 3000);
  }

  function unique(values) {
    return [...new Set(values.filter(Boolean))];
  }

  /* ------------------------------------------------------------- stratégie */

  function groupMembers(robots) {
    const groups = { main: [], hunter: [], swarm: [] };
    robots.forEach((robot) => {
      if (!groups[robot.role]) groups[robot.role] = [];
      groups[robot.role].push(robot);
    });
    return groups;
  }

  function stateFor(key) {
    return ((fleetData && fleetData.robots) || []).find((robot) => robot.key === key) || {};
  }

  function colorButtonsHtml(activeColor, disabled) {
    return [['1', 'BLEU', 'blue'], ['2', 'JAUNE', 'yellow']].map(([value, label, name]) => `
      <button class="color-btn${activeColor === name ? ' active' : ''}"
              data-color="${value}" ${disabled ? 'disabled' : ''}>${label}</button>`).join('');
  }

  function teamColor(robots) {
    const colors = unique(robots.map((robot) => stateFor(robot.key).color));
    return colors.length === 1 ? colors[0] : '';
  }

  function colorCard(robots) {
    const online = robots.some((robot) => stateFor(robot.key).online);
    const color = teamColor(robots);
    const label = color === 'blue' ? 'bleu' : color === 'yellow' ? 'jaune' : '—';
    return `
      <article class="card strategy-card color-card span-all" data-target="all">
        <div class="robot-head">
          <div>
            <div class="robot-title">Couleur de l'équipe</div>
            <div class="robot-sub">Commune au principal, au chasseur et à l'essaim</div>
          </div>
          <span class="pill ${online ? 'online' : 'offline'}">${label}</span>
        </div>
        <div class="color-row">${colorButtonsHtml(color, !online)}</div>
      </article>`;
  }

  function hostRowHtml(robot) {
    const state = stateFor(robot.key);
    const host = state.host || '';
    const port = state.port || 80;
    const roleLabel = robot.role === 'swarm'
      ? `${GROUP_TITLES.swarm} ${Number.isInteger(robot.index) ? robot.index + 1 : ''}`.trim()
      : (GROUP_TITLES[robot.role] || robot.role);
    return `
      <div class="host-row" data-key="${escapeHtml(robot.key)}">
        <span class="host-name">${escapeHtml(roleLabel)}</span>
        <input class="host-input" type="text" value="${escapeHtml(host)}"
               placeholder="192.168.1.10" spellcheck="false" autocomplete="off"
               aria-label="adresse IP ${escapeHtml(robot.name)}">
        <input class="port-input" type="number" min="1" max="65535" value="${port}"
               aria-label="port ${escapeHtml(robot.name)}">
        <button class="btn host-save">Enregistrer</button>
      </div>`;
  }

  function hostsCard(robots) {
    if (!robots.length) return '';
    return `
      <article class="card strategy-card hosts-card span-all">
        <div class="robot-head">
          <div>
            <div class="robot-title">Adresses des robots</div>
            <div class="robot-sub">Modifiables à chaud et enregistrées dans le fichier de configuration</div>
          </div>
        </div>
        <div class="host-list">${robots.map(hostRowHtml).join('')}</div>
      </article>`;
  }

  function strategyCard(role, members) {
    const states = members.map((robot) => stateFor(robot.key));
    const online = states.some((state) => state.online);
    const strategies = unique(members.flatMap((robot) => robot.strategies || []));
    const current = unique(states.map((state) => state.strategy));
    const groupStrategy = current.length === 1 ? current[0] : '';

    const strategyButtons = strategies.length
      ? strategies.map((name) => `
        <button class="strat-btn${name === groupStrategy ? ' active' : ''}"
                data-strat="${escapeHtml(name)}" ${online ? '' : 'disabled'}>${escapeHtml(name)}</button>`).join('')
      : '<span class="muted small">Aucune stratégie disponible</span>';

    const stateLines = members.map((member) => {
      const state = stateFor(member.key);
      const detail = state.strategy ? `stratégie : ${state.strategy}` : 'stratégie ?';
      return `
        <div class="state-line">
          <span>${escapeHtml(member.name)} <span class="muted">— ${escapeHtml(detail)}</span></span>
          <span class="pill ${state.online ? 'online' : 'offline'}">${state.online ? 'en ligne' : 'hors ligne'}</span>
        </div>`;
    }).join('');

    const host = members.length === 1
      ? `${members[0].name}${stateFor(members[0].key).host ? ' · ' + escapeHtml(stateFor(members[0].key).host) : ''}`
      : `${members.length} robots · même stratégie`;

    return `
      <article class="card strategy-card" data-target="${role}">
        <div class="robot-head">
          <div>
            <div class="robot-title">${GROUP_TITLES[role]}</div>
            <div class="robot-sub">${host}</div>
          </div>
          <span class="pill ${online ? 'online' : 'offline'}">${online ? 'en ligne' : 'hors ligne'}</span>
        </div>
        <p class="strat-label">Stratégie</p>
        <div class="strat-row">${strategyButtons}</div>
        <div class="robot-state">${stateLines}</div>
      </article>`;
  }

  function renderStrategy() {
    const grid = $('strategy-grid');
    if (!grid) return;
    const robots = (fleetData && fleetData.robots) || [];
    const groups = groupMembers(robots);
    const cards = [];
    if (robots.length) cards.push(colorCard(robots));
    GROUPS.filter((role) => groups[role] && groups[role].length)
      .forEach((role) => cards.push(strategyCard(role, groups[role])));
    if (robots.length) cards.push(hostsCard(robots));
    grid.innerHTML = cards.length
      ? cards.join('')
      : '<p class="muted">Aucun robot configuré. Renseignez la section <code>robots</code> de config/default.json.</p>';
  }

  async function loadStrategy(force = false) {
    try {
      const [fleet, strategies] = await Promise.all([api('/fleet'), api('/fleet/strategies')]);
      fleetData = { ...fleet, robots: strategies.robots.map((robot) => ({
        ...robot, ...(fleet.robots.find((item) => item.key === robot.key) || {}),
      })) };
      const signature = JSON.stringify([fleet.robots, strategies.robots]);
      if (force || signature !== strategySignature) {
        strategySignature = signature;
        renderStrategy();
      }
    } catch (error) {
      strategySignature = '';
      fleetToast(`Flotte indisponible : ${error.message}`, true);
      const grid = $('strategy-grid');
      if (grid && !fleetData) {
        grid.innerHTML = `<p class="muted">Flotte indisponible : ${escapeHtml(error.message)}</p>`;
      }
    }
  }

  async function onStrategyClick(event) {
    const card = event.target.closest('.strategy-card');
    if (!card) return;

    const saveButton = event.target.closest('.host-save');
    if (saveButton) {
      const row = saveButton.closest('.host-row');
      const key = row.dataset.key;
      const host = row.querySelector('.host-input').value.trim();
      const port = Number(row.querySelector('.port-input').value) || 80;
      saveButton.disabled = true;
      try {
        await post(`/fleet/${encodeURIComponent(key)}/host`, { host, port });
        fleetToast('Adresse enregistrée');
        await loadStrategy(true);
      } catch (error) {
        saveButton.disabled = false;
        fleetToast(error.message, true);
      }
      return;
    }

    const target = card.dataset.target;
    const colorButton = event.target.closest('.color-btn');
    const strategyButton = event.target.closest('.strat-btn');
    try {
      if (colorButton) {
        await post(`/fleet/${encodeURIComponent(target)}/color`, { color: Number(colorButton.dataset.color) });
        fleetToast('Couleur envoyée');
        await loadStrategy(true);
      } else if (strategyButton) {
        await post(`/fleet/${encodeURIComponent(target)}/strategy`, { strat: strategyButton.dataset.strat });
        fleetToast('Stratégie envoyée');
        await loadStrategy(true);
      }
    } catch (error) {
      fleetToast(error.message, true);
    }
  }

  /* ------------------------------------------------------------ table live */

  function ensureMap(source) {
    const src = source || MAP_DEFAULT;
    if (mapImage && mapSource === src) return;
    mapSource = src;
    const image = new Image();
    image.onload = () => { mapImage = image; if (livePayload) drawLive(livePayload); };
    image.onerror = () => { mapImage = null; };
    image.src = src;
  }

  function drawLive(live) {
    const canvas = $('live-canvas');
    if (!canvas) return;
    const table = live.table || {};
    const widthMm = table.width_mm || 2000;
    const heightMm = table.height_mm || 3000;
    canvas.style.aspectRatio = `${widthMm} / ${heightMm}`;

    const ratio = window.devicePixelRatio || 1;
    const cssWidth = canvas.clientWidth;
    if (!cssWidth) return;
    const cssHeight = cssWidth * (heightMm / widthMm);
    if (canvas.width !== Math.round(cssWidth * ratio) || canvas.height !== Math.round(cssHeight * ratio)) {
      canvas.width = Math.round(cssWidth * ratio);
      canvas.height = Math.round(cssHeight * ratio);
    }

    const ctx = canvas.getContext('2d');
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, cssWidth, cssHeight);

    const padding = 10;
    const scale = Math.min(
      (cssWidth - 2 * padding) / widthMm,
      (cssHeight - 2 * padding) / heightMm
    );
    const centreX = cssWidth / 2;
    const centreY = cssHeight / 2;
    const sx = (x) => centreX + x * scale;
    const sy = (y) => centreY - y * scale;

    const left = sx(-widthMm / 2);
    const top = sy(heightMm / 2);
    const tableWidth = widthMm * scale;
    const tableHeight = heightMm * scale;

    ctx.fillStyle = '#0f1620';
    ctx.fillRect(left, top, tableWidth, tableHeight);
    if (mapImage) {
      try { ctx.drawImage(mapImage, left, top, tableWidth, tableHeight); } catch (_) { /* image invalide */ }
    }
    ctx.strokeStyle = '#2a3240';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(left, top, tableWidth, tableHeight);

    drawArrow(ctx, sx(0), sy(0), sx(500), sy(0), 'rgba(248,81,73,0.5)');
    drawArrow(ctx, sx(0), sy(0), sx(0), sy(500), 'rgba(88,166,255,0.5)');

    (live.objects || []).forEach((object) => drawGameObject(ctx, object, sx, sy));
    (live.robots || []).forEach((robot) => drawPath(ctx, robot, sx, sy));
    (live.opponents || []).forEach((opponent) => drawPath(ctx, opponent, sx, sy));

    (live.robots || []).forEach((robot) => {
      if (robot.x === null || robot.x === undefined) return;
      drawRobot(ctx, robot, sx, sy, ROBOT_COLORS[robot.color] || ROBOT_COLORS['']);
    });
    const opponentColor = ROBOT_COLORS[live.opponent_color] || ROBOT_COLORS[''];
    (live.opponents || []).forEach((opponent) => {
      drawOpponent(ctx, opponent, sx(opponent.x), sy(opponent.y), opponentColor);
    });
  }

  function drawPath(ctx, item, sx, sy) {
    const path = item.path || [];
    if (path.length < 2) return;
    ctx.strokeStyle = 'rgba(139, 151, 168, 0.55)';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.moveTo(sx(path[0][0]), sy(path[0][1]));
    for (let index = 1; index < path.length; index += 1) {
      ctx.lineTo(sx(path[index][0]), sy(path[index][1]));
    }
    ctx.stroke();
  }

  function drawRobot(ctx, robot, sx, sy, color) {
    const x = sx(robot.x);
    const y = sy(robot.y);
    ctx.save();
    if (robot.role === 'main') {
      ctx.fillStyle = color;
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y, 11, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      orient(ctx, x, y, robot.a);
    } else if (robot.role === 'hunter') {
      ctx.fillStyle = color;
      ctx.strokeStyle = '#0d1117';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x, y - 11);
      ctx.lineTo(x + 10, y + 8);
      ctx.lineTo(x - 10, y + 8);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    } else {
      ctx.fillStyle = color;
      ctx.strokeStyle = '#0d1117';
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(x, y, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
    ctx.fillStyle = '#e6edf3';
    ctx.font = '10px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(robot.name, x, y - 16);
    ctx.restore();
  }

  function drawOpponent(ctx, opponent, x, y, color) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x, y, 11, 0, Math.PI * 2);
    ctx.stroke();
    orient(ctx, x, y, opponent.a);
    ctx.fillStyle = color;
    ctx.font = '10px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`adverse #${opponent.id}`, x, y - 17);
    ctx.restore();
  }

  function orient(ctx, x, y, angleDeg) {
    const angle = ((angleDeg || 0) * Math.PI) / 180;
    ctx.strokeStyle = '#e6edf3';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + Math.cos(angle) * 18, y - Math.sin(angle) * 18);
    ctx.stroke();
  }

  function drawGameObject(ctx, object, sx, sy) {
    const x = sx(object.x);
    const y = sy(object.y);
    const angle = ((object.a || 0) * Math.PI) / 180;
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(-angle);
    ctx.fillStyle = 'rgba(63, 185, 80, 0.85)';
    ctx.strokeStyle = '#e6edf3';
    ctx.lineWidth = 1;
    ctx.fillRect(-9, -6, 18, 12);
    ctx.strokeRect(-9, -6, 18, 12);
    ctx.restore();
  }

  function drawArrow(ctx, fromX, fromY, toX, toY, color) {
    const angle = Math.atan2(toY - fromY, toX - fromX);
    const head = 6;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.6;
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

  function positionText(robot) {
    if (robot.x === null || robot.x === undefined) return 'position inconnue';
    return `x ${Number(robot.x).toFixed(0)} · y ${Number(robot.y).toFixed(0)} · a ${Number(robot.a || 0).toFixed(0)}°`;
  }

  function renderLive(live) {
    drawLive(live);
    ensureMap(live.map_image);

    const updated = $('live-updated');
    if (updated) updated.textContent = new Date().toLocaleTimeString('fr-FR');

    const fleet = $('live-fleet');
    if (fleet) {
      fleet.innerHTML = (live.robots || []).map((robot) => `
        <div class="live-fleet-row">
          <span><i class="swatch" style="background:${ROBOT_COLORS[robot.color] || ROBOT_COLORS['']}"></i>
            ${escapeHtml(robot.name)} <span class="muted small">(${robot.source === 'camera' ? 'caméra' : 'déclaré'})</span></span>
          <span class="muted small">${positionText(robot)}</span>
        </div>`).join('') || '<span class="muted small">—</span>';
    }

    const opponents = $('live-opponents');
    if (opponents) {
      opponents.innerHTML = (live.opponents || []).length
        ? (live.opponents || []).map((opponent) => `
            <div class="live-fleet-row">
              <span>tag #${opponent.id}</span>
              <span class="muted small">x ${Number(opponent.x).toFixed(0)} · y ${Number(opponent.y).toFixed(0)}</span>
            </div>`).join('')
        : `Aucun (équipe ${live.opponent_color || '?'} non détectée)`;
    }

    const objects = $('live-objects');
    if (objects) {
      objects.innerHTML = (live.objects || []).length
        ? (live.objects || []).map((object) => `
            <div class="live-fleet-row">
              <span>${escapeHtml(object.label || '')} <span class="tag-chip">${object.id}</span></span>
              <span class="muted small">x ${Number(object.x).toFixed(0)} · y ${Number(object.y).toFixed(0)}</span>
            </div>`).join('')
        : 'Aucun objet détecté';
    }

    const empty = $('live-empty');
    if (empty) {
      const hasRobots = (live.robots || []).some((robot) => robot.x !== null && robot.x !== undefined);
      const hasObjects = (live.objects || []).length > 0 || (live.opponents || []).length > 0;
      empty.hidden = hasRobots || hasObjects;
    }
  }

  async function liveTick() {
    try {
      livePayload = await api('/fleet/live');
      renderLive(livePayload);
    } catch (_) { /* on retente au prochain tick */ }
  }

  /* ---------------------------------------------------------- boucles/état */

  function startStrategyPolling() {
    if (strategyTimer) return;
    loadStrategy();
    strategyTimer = setInterval(loadStrategy, STRATEGY_POLL_MS);
  }

  function stopStrategyPolling() {
    if (strategyTimer) { clearInterval(strategyTimer); strategyTimer = null; }
  }

  function startLivePolling() {
    if (liveTimer) return;
    liveTick();
    liveTimer = setInterval(liveTick, LIVE_POLL_MS);
  }

  function stopLivePolling() {
    if (liveTimer) { clearInterval(liveTimer); liveTimer = null; }
  }

  function onTabChange(tab) {
    currentTab = tab;
    if (tab === 'strategy') startStrategyPolling(); else stopStrategyPolling();
    if (tab === 'live') startLivePolling(); else stopLivePolling();
  }

  function wireUp() {
    const grid = $('strategy-grid');
    if (grid) grid.addEventListener('click', onStrategyClick);
    document.addEventListener('tabchange', (event) => onTabChange(event.detail.tab));
    window.addEventListener('resize', () => { if (livePayload) drawLive(livePayload); });
  }

  document.addEventListener('DOMContentLoaded', () => {
    wireUp();
    onTabChange(window.matTabs ? window.matTabs.current() : 'strategy');
  });
})();
