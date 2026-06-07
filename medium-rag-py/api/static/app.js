const ta = document.getElementById('question');

ta.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendQuery(); }
});

function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 200) + 'px';
}

function fillQ(btn) {
  ta.value = btn.textContent.trim();
  ta.focus();
  autoResize(ta);
}

function uploadLogo(input) {
  const file = input.files[0];
  if (!file) return;
  const img = document.getElementById('logoImg');
  const icon = document.getElementById('logoIcon');
  img.src = URL.createObjectURL(file);
  img.style.display = 'block';
  icon.style.display = 'none';
}

async function sendQuery() {
  const q = ta.value.trim();
  if (!q) return;

  const btn     = document.getElementById('sendBtn');
  const spinner = document.getElementById('spinner');
  const label   = document.getElementById('sendLabel');
  const icon    = document.getElementById('sendIcon');

  btn.disabled = true;
  spinner.style.display = 'block';
  icon.style.display    = 'none';
  label.textContent     = 'Thinking…';

  document.getElementById('result').style.display = 'none';

  try {
    const res  = await fetch('/api/prompt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: q }),
    });
    const data = await res.json();
    data.error
      ? showError(data.error + (data.detail ? ': ' + data.detail : ''))
      : renderResult(data);
  } catch (err) {
    showError('Network error: ' + err.message);
  } finally {
    btn.disabled          = false;
    spinner.style.display = 'none';
    icon.style.display    = '';
    label.textContent     = 'Ask';
  }
}

function renderResult(data) {
  document.getElementById('answerText').textContent  = data.response || '—';
  document.getElementById('answerCard').style.display = '';

  const ctx = data.context || [];
  document.getElementById('sourcesLabel').textContent =
    `Sources · ${ctx.length} chunk${ctx.length !== 1 ? 's' : ''} retrieved`;

  const grid = document.getElementById('sourcesGrid');
  grid.innerHTML = '';
  ctx.forEach(c => {
    const el = document.createElement(c.url ? 'a' : 'div');
    el.className = 'source-item';
    if (c.url) {
      el.href = c.url;
      el.target = '_blank';
      el.rel = 'noopener noreferrer';
    }
    el.innerHTML =
      `<div class="source-item-title">${esc(c.title || 'Untitled')}</div>` +
      (c.authors ? `<div class="source-item-author">${esc(c.authors)}</div>` : '') +
      `<div class="score-badge">` +
        `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">` +
          `<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>` +
        `</svg>` +
        (typeof c.score === 'number' ? c.score.toFixed(4) : c.score) +
      `</div>` +
      `<div class="source-snippet">${esc(c.chunk || '')}</div>`;
    grid.appendChild(el);
  });

  document.getElementById('result').style.display = 'block';
  document.getElementById('result').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function showError(msg) {
  const card = document.getElementById('answerCard');
  card.className = 'error-card';
  card.innerHTML = esc(msg);
  document.getElementById('sourcesCard').style.display = 'none';
  document.getElementById('result').style.display      = 'block';
}

function esc(s) {
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// Load stats on page load
(async () => {
  try {
    const s = await fetch('/api/stats').then(r => r.json());
    document.getElementById('sChunk').textContent   = s.chunk_size + ' tok';
    document.getElementById('sOverlap').textContent = s.overlap_ratio;
    document.getElementById('sTopK').textContent    = s.top_k;
    document.getElementById('statusDot').classList.remove('offline');
  } catch {
    document.getElementById('statusDot').classList.add('offline');
  }
})();
