'use strict';

/**
 * Onglet « Robot principal » : reprend les onglets de l'interface web du robot
 * dans des iframes pointant directement sur son adresse (le robot sert ses
 * propres pages). Les cadres sont chargés à la demande.
 */
(() => {
  const $ = (id) => document.getElementById(id);

  // Onglets de l'interface du robot (mêmes entrées que sa barre de navigation).
  const PAGES = [
    ['Accueil', '/'],
    ['Control', '/control'],
    ['Camera', '/camera'],
    ['Live Table', '/display'],
    ['Lidar', '/lidar'],
    ['PAMIs', '/pami'],
    ['Logs', '/logs'],
    ['Robot', '/robot'],
  ];

  let built = false;

  function select(path, base) {
    document.querySelectorAll('#robot-subtabs .tab-btn').forEach((button) => {
      button.classList.toggle('active', button.dataset.path === path);
    });
    document.querySelectorAll('#robot-frames .robot-frame').forEach((frame) => {
      const active = frame.dataset.path === path;
      if (active && !frame.getAttribute('src')) frame.setAttribute('src', base + frame.dataset.path);
      frame.hidden = !active;
    });
  }

  async function build() {
    if (built) return;
    built = true;

    const subtabs = $('robot-subtabs');
    const frames = $('robot-frames');
    const hint = $('robot-hint');
    if (!subtabs || !frames) return;

    let fleet;
    try {
      fleet = await fetch('/fleet', { cache: 'no-store' }).then((response) => response.json());
    } catch (error) {
      hint.textContent = `Flotte indisponible : ${error.message}`;
      return;
    }

    const main = ((fleet && fleet.robots) || []).find((robot) => robot.role === 'main');
    if (!main || !main.host) {
      hint.textContent = 'Adresse du robot principal non configurée (config/default.json → robots.main.host).';
      return;
    }

    const base = `http://${main.host}:${main.port || 80}`;
    hint.textContent = `${base} (le robot sert directement ses pages)`;

    subtabs.innerHTML = PAGES.map(([label, path], index) => `
      <button class="tab-btn${index === 0 ? ' active' : ''}" data-path="${path}">${label}</button>`).join('');
    frames.innerHTML = PAGES.map(([label, path], index) => `
      <iframe class="robot-frame" data-path="${path}" title="${label}"
              ${index === 0 ? `src="${base}${path}"` : ''} ${index === 0 ? '' : 'hidden'}></iframe>`).join('');

    subtabs.addEventListener('click', (event) => {
      const button = event.target.closest('.tab-btn');
      if (button) select(button.dataset.path, base);
    });
  }

  document.addEventListener('tabchange', (event) => {
    if (event.detail.tab === 'robot') build();
  });

  document.addEventListener('DOMContentLoaded', () => {
    if (window.matTabs && window.matTabs.current() === 'robot') build();
  });
})();
