// Tactus service worker: makes the site installable and lets the instrument open without a network.
// Pages come from the network first, so an update is seen on the next launch, and from the cache when offline.
// Fonts and the hand-tracking model come from other sites; they are kept after the first use (and refreshed in the
// background), so hands and skins still work offline once they have been used once.
const VERSION = '20261013134952';
const CORE = 'tactus-core-' + VERSION, EXTRA = 'tactus-extra';
const PRECACHE = ['./', './play/', './manifest.webmanifest', './favicon.svg', './favicon-32.png', './apple-touch-icon.png', './icon-192.png', './icon-512.png', './icon-maskable-512.png'];
const EXTRA_HOSTS = ['fonts.googleapis.com', 'fonts.gstatic.com', 'cdn.jsdelivr.net', 'storage.googleapis.com'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CORE).then(c => c.addAll(PRECACHE)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', e => {
  e.waitUntil(caches.keys().then(keys => Promise.all(keys.filter(k => k.startsWith('tactus-core-') && k !== CORE).map(k => caches.delete(k))))
    .then(() => self.clients.claim()));
});
self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.origin === location.origin) {
    if (!url.pathname.startsWith(new URL('./', location).pathname)) return;
    // network first; a page asked with ?gpu=off still finds the cached page
    e.respondWith(fetch(req).then(res => {
      if (res.ok) { const copy = res.clone(); caches.open(CORE).then(c => c.put(req, copy)); }
      return res;
    }).catch(() => caches.match(req, { ignoreSearch: true }).then(r => r || (req.mode === 'navigate' ? caches.match('./play/') : Response.error()))));
    return;
  }
  if (EXTRA_HOSTS.includes(url.hostname)) {
    e.respondWith(caches.open(EXTRA).then(async c => {
      const hit = await c.match(req);
      const net = fetch(req).then(res => { if (res.ok || res.type === 'opaque') c.put(req, res.clone()); return res; }).catch(() => hit);
      return hit || net;
    }));
  }
});
