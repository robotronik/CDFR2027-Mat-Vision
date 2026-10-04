'use strict';

/**
 * Bascule entre les onglets principaux de l'interface du mat.
 * Chaque changement émet un évènement `tabchange` sur `document` pour que les
 * autres scripts (vision, table live, robot) démarrent ou arrêtent leur boucle.
 */
(() => {
  const PANELS = {
    strategy: 'panel-strategy',
    live: 'panel-live',
    vision: 'panel-vision',
    robot: 'panel-robot',
  };

  const buttons = Array.from(document.querySelectorAll('#main-tabs .tab-btn'));
  let current = buttons.find((b) => b.classList.contains('active'))?.dataset.tab || 'strategy';

  function activate(tab) {
    if (!PANELS[tab]) return;
    current = tab;
    buttons.forEach((button) => button.classList.toggle('active', button.dataset.tab === tab));
    Object.entries(PANELS).forEach(([name, id]) => {
      const panel = document.getElementById(id);
      if (panel) panel.classList.toggle('active', name === tab);
    });
    document.dispatchEvent(new CustomEvent('tabchange', { detail: { tab } }));
  }

  buttons.forEach((button) =>
    button.addEventListener('click', () => activate(button.dataset.tab))
  );

  window.matTabs = { activate, current: () => current };
})();
