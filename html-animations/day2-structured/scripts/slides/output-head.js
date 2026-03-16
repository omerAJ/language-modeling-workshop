/* Slide 32 — From Hidden State to Next Word (Output Head) */

const PRED32_MAX_STEP = 4;
const PRED32_TOKENS = ['cat', 'sat', 'on', 'the'];
const PRED32_FOCUS_IDX = 3; // 'the' is the last token
const PRED32_VOCAB = [
  { token: 'mat', prob: 0.42 },
  { token: 'rug', prob: 0.24 },
  { token: '.', prob: 0.16 },
  { token: 'cat', prob: 0.11 },
  { token: 'sat', prob: 0.07 }
];
const PRED32_BAR_STAGGER_MS = 70;

const PRED32_CLASSES = [
  'pred32-show-extract',
  'pred32-show-wu',
  'pred32-show-bars',
  'pred32-show-contrast'
];

const PRED32_TAKEAWAYS = [
  'After L blocks, the residual matrix holds one contextualized row per token.',
  'To predict the next token, extract the last row \\u2014 it encodes the full left context.',
  'The LM head is a single linear projection scoring every token in the vocabulary.',
  'Softmax converts raw logits into probabilities. Weight tying reuses the embedding matrix.',
  'Two different softmaxes, two different roles \\u2014 don\\u2019t confuse them.'
];

function buildPred32PromptRow() {
  var row = document.getElementById('pred32-prompt-row');
  if (!row || row.children.length > 0) return;
  PRED32_TOKENS.forEach(function(tok) {
    row.appendChild(createEl('span', { className: 'attn19-token-chip', text: tok }));
  });
  row.appendChild(createEl('span', { className: 'pred32-next-label', html: '&#8594; predict next' }));
}

function buildPred32Matrix() {
  var zone = document.getElementById('pred32-matrix-zone');
  if (!zone || zone.children.length > 0) return;

  var card = createEl('div', { className: 'pred32-matrix-card' });
  card.appendChild(createEl('div', { className: 'head-card-title', html: '\\(R^{(L)} \\in \\mathbb{R}^{S \\times d_{\\mathrm{model}}}\\)' }));

  var matrix = createEl('div', { className: 'head-matrix' });
  PRED32_TOKENS.forEach(function(tok, i) {
    var label = '\\(h_{' + (i + 1) + '}\\)';
    var rowCls = 'head-row' + (i === PRED32_FOCUS_IDX ? ' is-focus' : '');
    matrix.appendChild(createEl('div', { className: rowCls }, [
      createEl('span', { html: label }),
      createEl('div', { className: 'head-bar' })
    ]));
  });
  card.appendChild(matrix);
  zone.appendChild(card);
}

function buildPred32Extract() {
  var zone = document.getElementById('pred32-extract-zone');
  if (!zone || zone.children.length > 0) return;
  zone.appendChild(createEl('div', { className: 'pred32-step-label', text: 'extract last row' }));
  zone.appendChild(createEl('div', { className: 'pred32-extract-chip', html: '\\(h_t^{(L)}\\)' }));
  zone.appendChild(createEl('div', { className: 'pred32-extract-note', text: 'every block was building this vector' }));
}

function buildPred32Wu() {
  var zone = document.getElementById('pred32-wu-zone');
  if (!zone || zone.children.length > 0) return;
  zone.appendChild(createEl('div', { className: 'pred32-step-label', text: 'LM Head' }));
  zone.appendChild(createEl('div', { className: 'pred32-formula', html: '\\(\\ell_t = W_U \\, h_t^{(L)}\\)' }));
  zone.appendChild(createEl('div', { className: 'pred32-dims', html: '\\(W_U \\in \\mathbb{R}^{|V| \\times d_{\\mathrm{model}}}\\)' }));
}

function buildPred32Bars() {
  var zone = document.getElementById('pred32-bars-zone');
  if (!zone || zone.children.length > 0) return;
  zone.appendChild(createEl('div', { className: 'pred32-softmax-label', text: 'softmax \u2192 probabilities' }));
  var barsRow = createEl('div', { className: 'pred32-bars-row' });
  PRED32_VOCAB.forEach(function(entry, i) {
    var pct = Math.round(entry.prob * 100);
    var barCls = 'pred32-bar' + (i === 0 ? ' is-top' : '');
    barsRow.appendChild(createEl('div', { className: 'pred32-bar-item' }, [
      createEl('strong', { text: pct + '%' }),
      createEl('div', { className: barCls, style: { height: '0%' }, dataset: { target: (entry.prob * 100) + '%' } }),
      createEl('span', { className: 'pred32-bar-label', text: entry.token })
    ]));
  });
  zone.appendChild(barsRow);
}

function buildPred32TyingNote() {
  var note = document.getElementById('pred32-tying-note');
  if (!note || note.children.length > 0) return;
  note.appendChild(createEl('div', { className: 'callout info' }, [
    createEl('span', { className: 'icon', html: '&#x1F517;' }),
    createEl('span', { html: 'In most models \\(W_U = W_E^\\top\\) \\u2014 the same matrix used for embeddings, transposed. This is called <strong>weight tying</strong>.' })
  ]));
}

function buildPred32Contrast() {
  var zone = document.getElementById('pred32-contrast');
  if (!zone || zone.children.length > 0) return;
  zone.appendChild(createEl('div', { className: 'callout question' }, [
    createEl('span', { className: 'icon', html: '&#x1F914;' }),
    createEl('span', { html: 'This softmax normalizes over \\(|V|\\) tokens (the vocabulary). How is that different from the attention softmax?' })
  ]));
  var grid = createEl('div', { className: 'pred32-contrast-grid' });
  grid.appendChild(createEl('div', { className: 'pred32-contrast-card' }, [
    createEl('div', { className: 'pred32-contrast-title hl-purple', text: 'Attention softmax' }),
    createEl('div', { className: 'pred32-contrast-body', text: 'Normalizes over positions \u2014 which tokens to attend to' })
  ]));
  grid.appendChild(createEl('div', { className: 'pred32-contrast-card' }, [
    createEl('div', { className: 'pred32-contrast-title hl-green', text: 'Output softmax' }),
    createEl('div', { className: 'pred32-contrast-body', text: 'Normalizes over vocabulary \u2014 which token to predict' })
  ]));
  zone.appendChild(grid);
}

function animatePred32Bars() {
  var bars = document.querySelectorAll('.pred32-bar');
  outputHeadState.timers.forEach(function(t) { clearTimeout(t); });
  outputHeadState.timers = [];
  bars.forEach(function(bar, i) {
    var tid = setTimeout(function() {
      bar.style.height = bar.dataset.target;
    }, i * PRED32_BAR_STAGGER_MS);
    outputHeadState.timers.push(tid);
  });
}

function resetPred32Bars() {
  var bars = document.querySelectorAll('.pred32-bar');
  bars.forEach(function(bar) {
    bar.style.transition = 'none';
    bar.style.height = '0%';
    void bar.offsetHeight; // force reflow
    bar.style.transition = '';
  });
}

function setOutputHeadStep(step) {
  var slide = document.getElementById('slide-32');
  var takeaway = document.getElementById('pred32-takeaway');
  if (!slide || !takeaway) return;

  var clamped = Math.max(0, Math.min(PRED32_MAX_STEP, step));
  outputHeadState.step = clamped;

  PRED32_CLASSES.forEach(function(cls, idx) {
    slide.classList.toggle(cls, clamped >= idx + 1);
  });

  if (clamped >= 3) {
    animatePred32Bars();
  } else {
    resetPred32Bars();
  }

  takeaway.innerHTML = PRED32_TAKEAWAYS[clamped] || PRED32_TAKEAWAYS[0];
  typesetMath(slide);
}

function initOutputHeadSlide() {
  if (!outputHeadState.initialized) {
    buildPred32PromptRow();
    buildPred32Matrix();
    buildPred32Extract();
    buildPred32Wu();
    buildPred32Bars();
    buildPred32TyingNote();
    buildPred32Contrast();
    outputHeadState.initialized = true;
  }
  setOutputHeadStep(outputHeadState.step || 0);
  typesetMath(document.getElementById('slide-32'));
}

function runOutputHeadStep() {
  if (outputHeadState.step >= PRED32_MAX_STEP) return false;
  setOutputHeadStep(outputHeadState.step + 1);
  return true;
}

function resetOutputHeadSlide() {
  outputHeadState.timers.forEach(function(t) { clearTimeout(t); });
  outputHeadState.timers = [];
  outputHeadState.step = 0;
  resetPred32Bars();
  var slide = document.getElementById('slide-32');
  if (slide) {
    PRED32_CLASSES.forEach(function(cls) { slide.classList.remove(cls); });
  }
  var takeaway = document.getElementById('pred32-takeaway');
  if (takeaway) takeaway.innerHTML = PRED32_TAKEAWAYS[0];
}
