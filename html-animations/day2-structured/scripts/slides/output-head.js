/* Slide 32 — Predict One Token, Append, Repeat */

var PRED32_MAX_STEP = 1;
var PRED32_TOKENS = ['The', 'cat', 'sat', 'on', 'the'];
var PRED32_VOCAB = [
  { token: 'mat',   prob: 0.42 },
  { token: 'rug',   prob: 0.24 },
  { token: '.',     prob: 0.16 },
  { token: 'floor', prob: 0.11 },
  { token: 'bed',   prob: 0.07 }
];
var PRED32_SELECTED = 'mat';
var PRED32_BAR_STAGGER_MS = 60;

var PRED32_CLASSES = ['pred32-show-loop'];

var PRED32_TAKEAWAYS = [
  'One forward pass, one token. The model never generates more than one token at a time.',
  'Pick the token, append it to context, and run the full forward pass again.'
];

function buildPred32Zones() {
  // Zone 1: context tokens + last-row label
  var ctxZone = document.getElementById('pred32-ctx-zone');
  if (ctxZone && ctxZone.children.length === 0) {
    var tokensRow = createEl('div', { className: 'pred32-tokens-row' });
    PRED32_TOKENS.forEach(function(tok) {
      tokensRow.appendChild(createEl('span', { className: 'attn19-token-chip', text: tok }));
    });
    ctxZone.appendChild(tokensRow);
    ctxZone.appendChild(createEl('div', {
      className: 'pred32-zone-note',
      html: '\\(h_t^{(L)}\\) = last row of \\(R^{(L)}\\)'
    }));
  }

  // Zone 2: LM head formula + bars
  var headZone = document.getElementById('pred32-head-zone');
  if (headZone && headZone.children.length === 0) {
    headZone.appendChild(createEl('div', { className: 'pred32-zone-title', text: 'LM Head' }));
    headZone.appendChild(createEl('div', {
      className: 'pred32-formula',
      html: '\\(\\ell = W_U\\, h_t^{(L)}\\)'
    }));
    headZone.appendChild(createEl('div', {
      className: 'pred32-softmax-label',
      text: 'softmax \u2192 probabilities'
    }));
    var barsRow = createEl('div', { className: 'pred32-bars-row' });
    PRED32_VOCAB.forEach(function(entry, i) {
      var pct = Math.round(entry.prob * 100);
      var barCls = 'pred32-bar' + (i === 0 ? ' is-top' : '');
      barsRow.appendChild(createEl('div', { className: 'pred32-bar-item' }, [
        createEl('strong', { text: pct + '%' }),
        createEl('div', {
          className: barCls,
          style: { height: '0' },
          dataset: { target: (entry.prob * 5.0) + 'rem' }
        }),
        createEl('span', { className: 'pred32-bar-label', text: entry.token })
      ]));
    });
    headZone.appendChild(barsRow);
  }

  // Zone 3: selected token + loop annotation
  var selectZone = document.getElementById('pred32-select-zone');
  if (selectZone && selectZone.children.length === 0) {
    selectZone.appendChild(createEl('div', {
      className: 'pred32-zone-title pred32-zone-title-green',
      text: 'Next Token'
    }));
    selectZone.appendChild(createEl('span', {
      className: 'attn19-token-chip pred32-selected-token',
      text: PRED32_SELECTED
    }));
    selectZone.appendChild(createEl('div', {
      className: 'pred32-loop-note',
      html: '&#x21A9; append &amp; repeat'
    }));
  }
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
    bar.style.height = '0';
    void bar.offsetHeight;
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

  var tid = setTimeout(function() { animatePred32Bars(); }, 80);
  outputHeadState.timers.push(tid);

  takeaway.innerHTML = PRED32_TAKEAWAYS[clamped] || PRED32_TAKEAWAYS[0];
  typesetMath(slide);
}

function initOutputHeadSlide() {
  if (!outputHeadState.initialized) {
    buildPred32Zones();
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
