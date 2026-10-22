'use strict';

/**
 * Utilitaires partagés par les scripts de l'interface (scripts classiques,
 * sans bundler : l'objet est exposé sur `window.matvision`).
 */
window.matvision = (() => {
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

  function post(path, payload) {
    return api(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
  }

  function num(value, digits = 1) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
    return Number(value).toFixed(digits);
  }

  function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>"']/g, (char) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    })[char]);
  }

  function toast(message, isError = false, elementId = 'toast') {
    const element = document.getElementById(elementId);
    if (!element) return;
    element.textContent = message;
    element.style.color = isError ? 'var(--danger)' : 'var(--accent)';
    element.hidden = false;
    clearTimeout(element._timer);
    element._timer = setTimeout(() => { element.hidden = true; }, 3000);
  }

  function drawArrow(ctx, fromX, fromY, toX, toY, color, head = 7, lineWidth = 1.8) {
    const angle = Math.atan2(toY - fromY, toX - fromX);
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
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

  return { api, post, num, escapeHtml, toast, drawArrow };
})();
