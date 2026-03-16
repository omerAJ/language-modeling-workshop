/* Slide 33 — Generate, Append, Repeat */

var GEN33_MAX_STEP = 5;
var GEN33_INITIAL_CONTEXT = ['The', 'cat', 'sat', 'on', 'the'];
var GEN33_PASSES = [
  {
    dist: [
      { token: 'mat', prob: 0.42, logit: 3.2 },
      { token: 'rug', prob: 0.24, logit: 2.1 },
      { token: '.', prob: 0.16, logit: 1.4 },
      { token: 'floor', prob: 0.11, logit: 0.8 },
      { token: 'bed', prob: 0.07, logit: 0.3 }
    ],
    selected: 'mat'
  },
  {
    dist: [
      { token: '.', prob: 0.51, logit: 3.8 },
      { token: ',', prob: 0.22, logit: 2.0 },
      { token: 'and', prob: 0.14, logit: 1.2 },
      { token: '!', prob: 0.08, logit: 0.6 },
      { token: 'with', prob: 0.05, logit: 0.1 }
    ],
    selected: '.'
  }
];
var GEN33_BAR_STAGGER_MS = 50;
var GEN33_PHASE_MS = 400;

var GEN33_CLASSES = [
  'gen33-show-pass1',
  'gen33-show-loop',
  'gen33-show-strategies',
  'gen33-show-temp',
  'gen33-show-recap'
];

var GEN33_TAKEAWAYS = [
  'Generation starts from a prompt. The model predicts one token at a time.',
  'Forward pass produces a distribution. Pick a token and append it to the context.',
  'Autoregressive: predict one token, append, run the full forward pass again.',
  'How do we pick from the distribution? Three common decoding strategies.',
  'Temperature controls randomness. Low = focused, high = diverse.',
  'You now know every component from raw text to generated output.'
];

function applyTemperature(logits, T) {
  var clamped = Math.max(0.01, T);
  var scaled = logits.map(function(l) { return l / clamped; });
  var maxS = Math.max.apply(null, scaled);
  var exps = scaled.map(function(s) { return Math.exp(s - maxS); });
  var sum = exps.reduce(function(a, b) { return a + b; }, 0);
  return exps.map(function(e) { return e / sum; });
}

function buildGen33Context() {
  var ctx = document.getElementById('gen33-context');
  if (!ctx) return;
  ctx.innerHTML = '';
  var tokens = GEN33_INITIAL_CONTEXT.slice();
  if (generationState.step >= 1) tokens.push(GEN33_PASSES[0].selected);
  if (generationState.step >= 2) tokens.push(GEN33_PASSES[1].selected);
  tokens.forEach(function(tok, i) {
    var isNew = (generationState.step === 1 && i === GEN33_INITIAL_CONTEXT.length) ||
                (generationState.step === 2 && i === GEN33_INITIAL_CONTEXT.length + 1);
    var cls = 'attn19-token-chip' + (isNew ? ' gen33-token-new' : '');
    ctx.appendChild(createEl('span', { className: cls, text: tok }));
  });
}

function buildGen33Bars(passIdx) {
  var panel = document.getElementById('gen33-dist-panel');
  if (!panel) return;
  panel.innerHTML = '';
  var pass = GEN33_PASSES[passIdx] || GEN33_PASSES[0];
  pass.dist.forEach(function(entry) {
    var pct = Math.round(entry.prob * 100);
    var isSelected = entry.token === pass.selected;
    var barCls = 'gen33-bar' + (isSelected ? ' is-selected' : '');
    panel.appendChild(createEl('div', { className: 'gen33-bar-item' }, [
      createEl('strong', { text: pct + '%' }),
      createEl('div', { className: barCls, style: { height: '0%' }, dataset: { target: (entry.prob * 100) + '%', logit: '' + entry.logit } }),
      createEl('span', { className: 'gen33-bar-label', text: entry.token })
    ]));
  });
}

function animateGen33Bars() {
  var bars = document.querySelectorAll('#gen33-dist-panel .gen33-bar');
  generationState.timers.forEach(function(t) { clearTimeout(t); });
  generationState.timers = [];
  bars.forEach(function(bar, i) {
    var tid = setTimeout(function() {
      bar.style.height = bar.dataset.target;
    }, i * GEN33_BAR_STAGGER_MS);
    generationState.timers.push(tid);
  });
}

function resetGen33Bars() {
  var bars = document.querySelectorAll('#gen33-dist-panel .gen33-bar');
  bars.forEach(function(bar) {
    bar.style.transition = 'none';
    bar.style.height = '0%';
    void bar.offsetHeight;
    bar.style.transition = '';
  });
}

function buildGen33Selected(passIdx) {
  var sel = document.getElementById('gen33-selected');
  if (!sel) return;
  sel.innerHTML = '';
  var pass = GEN33_PASSES[passIdx] || GEN33_PASSES[0];
  sel.appendChild(createEl('span', { className: 'attn19-token-chip gen33-new-token', text: pass.selected }));
}

function buildGen33Strategies() {
  var panel = document.getElementById('gen33-strategies-panel');
  if (!panel || panel.children.length > 0) return;
  var strategies = [
    {
      title: 'Greedy',
      body: 'Always pick the top token. Deterministic, but tends to produce repetitive outputs.',
      formula: '\\(x_{t+1} = \\arg\\max_v\\, p_v\\)'
    },
    {
      title: 'Temperature \\(T\\)',
      body: 'Divide logits by \\(T\\) before softmax. Low \\(T\\) \\u2192 focused; high \\(T\\) \\u2192 diverse.',
      formula: '\\(p_v \\propto \\exp(\\ell_v / T)\\)',
      hasSlider: true
    },
    {
      title: 'Top-\\(p\\) (Nucleus)',
      body: 'Keep only tokens whose cumulative probability reaches \\(p\\). Sample from that set.',
      formula: 'sample from \\(\\{v : \\sum p_{v\'} \\ge p\\}\\)'
    }
  ];
  strategies.forEach(function(s) {
    var card = createEl('div', { className: 'gen33-strategy' });
    card.appendChild(createEl('div', { className: 'gen33-strategy-title', html: s.title }));
    card.appendChild(createEl('div', { className: 'gen33-strategy-body', text: s.body }));
    card.appendChild(createEl('div', { className: 'gen33-strategy-formula', html: s.formula }));
    if (s.hasSlider) {
      var wrap = createEl('div', { className: 'gen33-temp-wrap' });
      var slider = createEl('input', {
        type: 'range',
        id: 'gen33-temp-slider',
        min: '0.1',
        max: '3.0',
        step: '0.1',
        value: '1.0',
        className: 'gen33-temp-slider'
      });
      var label = createEl('span', { id: 'gen33-temp-value', className: 'gen33-temp-value', text: 'T = 1.0' });
      wrap.appendChild(slider);
      wrap.appendChild(label);
      card.appendChild(wrap);
    }
    panel.appendChild(card);
  });
}

function buildGen33Recap() {
  var recap = document.getElementById('gen33-pipeline-recap');
  if (!recap || recap.children.length > 0) return;
  var segments = [
    { text: 'embed', cls: 'gen33-seg-embed' },
    { text: '\u2192', cls: 'gen33-seg-arrow' },
    { text: 'position', cls: 'gen33-seg-position' },
    { text: '\u2192', cls: 'gen33-seg-arrow' },
    { text: '[', cls: 'gen33-seg-bracket' },
    { text: 'LN', cls: 'gen33-seg-norm' },
    { text: '\u2192', cls: 'gen33-seg-arrow-sm' },
    { text: 'attn', cls: 'gen33-seg-attn' },
    { text: '\u2192', cls: 'gen33-seg-arrow-sm' },
    { text: '+', cls: 'gen33-seg-add' },
    { text: '\u2192', cls: 'gen33-seg-arrow-sm' },
    { text: 'LN', cls: 'gen33-seg-norm' },
    { text: '\u2192', cls: 'gen33-seg-arrow-sm' },
    { text: 'FFN', cls: 'gen33-seg-ffn' },
    { text: '\u2192', cls: 'gen33-seg-arrow-sm' },
    { text: '+', cls: 'gen33-seg-add' },
    { text: '] \u00d7 L', cls: 'gen33-seg-bracket' },
    { text: '\u2192', cls: 'gen33-seg-arrow' },
    { text: 'LM head', cls: 'gen33-seg-head' },
    { text: '\u2192', cls: 'gen33-seg-arrow' },
    { text: 'decode', cls: 'gen33-seg-decode' }
  ];
  segments.forEach(function(seg) {
    recap.appendChild(createEl('span', { className: 'gen33-seg ' + seg.cls, text: seg.text }));
  });
}

function buildGen33Question() {
  var q = document.getElementById('gen33-question');
  if (!q || q.children.length > 0) return;
  q.appendChild(createEl('div', { className: 'callout question' }, [
    createEl('span', { className: 'icon', html: '&#x1F914;' }),
    createEl('span', { text: 'For code generation, low or high temperature? What about brainstorming new research directions?' })
  ]));
}

function bindGen33Slider() {
  if (generationState.sliderBound) return;
  var slider = document.getElementById('gen33-temp-slider');
  if (!slider) return;
  addTrackedListener(slider, 'input', function() {
    var T = parseFloat(slider.value);
    var label = document.getElementById('gen33-temp-value');
    if (label) label.textContent = 'T = ' + T.toFixed(1);
    var logits = GEN33_PASSES[0].dist.map(function(d) { return d.logit; });
    var newProbs = applyTemperature(logits, T);
    var maxProb = Math.max.apply(null, newProbs);
    var bars = document.querySelectorAll('#gen33-dist-panel .gen33-bar');
    bars.forEach(function(bar, i) {
      if (i < newProbs.length) {
        var pct = (newProbs[i] / maxProb) * 85;
        bar.style.height = pct + '%';
        var strong = bar.parentElement.querySelector('strong');
        if (strong) strong.textContent = Math.round(newProbs[i] * 100) + '%';
      }
    });
  });
  generationState.sliderBound = true;
}

function setGenerationStep(step) {
  var slide = document.getElementById('slide-33');
  var takeaway = document.getElementById('gen33-takeaway');
  if (!slide || !takeaway) return;

  var clamped = Math.max(0, Math.min(GEN33_MAX_STEP, step));
  generationState.step = clamped;

  GEN33_CLASSES.forEach(function(cls, idx) {
    slide.classList.toggle(cls, clamped >= idx + 1);
  });

  // Build context tokens based on current step
  buildGen33Context();

  // Build appropriate distribution bars
  if (clamped >= 2) {
    buildGen33Bars(1);
    buildGen33Selected(1);
  } else if (clamped >= 1) {
    buildGen33Bars(0);
    buildGen33Selected(0);
  }

  if (clamped >= 1) {
    // slight delay then animate bars
    var tid = setTimeout(function() { animateGen33Bars(); }, 80);
    generationState.timers.push(tid);
  } else {
    resetGen33Bars();
  }

  if (clamped >= 3) {
    buildGen33Strategies();
  }

  if (clamped >= 4) {
    bindGen33Slider();
    // Reset slider to 1.0 and rebuild pass-0 bars for interactive mode
    var slider = document.getElementById('gen33-temp-slider');
    if (slider) {
      slider.value = '1.0';
      var label = document.getElementById('gen33-temp-value');
      if (label) label.textContent = 'T = 1.0';
    }
    buildGen33Bars(0);
    var tid2 = setTimeout(function() { animateGen33Bars(); }, 80);
    generationState.timers.push(tid2);
  }

  if (clamped >= 5) {
    buildGen33Recap();
    buildGen33Question();
  }

  takeaway.innerHTML = GEN33_TAKEAWAYS[clamped] || GEN33_TAKEAWAYS[0];
  typesetMath(slide);
}

function initGenerationSlide() {
  if (!generationState.initialized) {
    generationState.initialized = true;
  }
  setGenerationStep(generationState.step || 0);
  typesetMath(document.getElementById('slide-33'));
}

function runGenerationStep() {
  if (generationState.step >= GEN33_MAX_STEP) return false;
  setGenerationStep(generationState.step + 1);
  return true;
}

function resetGenerationSlide() {
  generationState.timers.forEach(function(t) { clearTimeout(t); });
  generationState.timers = [];
  generationState.step = 0;
  var slide = document.getElementById('slide-33');
  if (slide) {
    GEN33_CLASSES.forEach(function(cls) { slide.classList.remove(cls); });
  }

  // Reset slider
  var slider = document.getElementById('gen33-temp-slider');
  if (slider) {
    slider.value = '1.0';
    var label = document.getElementById('gen33-temp-value');
    if (label) label.textContent = 'T = 1.0';
  }

  // Clear dynamic content
  var ctx = document.getElementById('gen33-context');
  if (ctx) ctx.innerHTML = '';
  var dist = document.getElementById('gen33-dist-panel');
  if (dist) dist.innerHTML = '';
  var sel = document.getElementById('gen33-selected');
  if (sel) sel.innerHTML = '';
  var strat = document.getElementById('gen33-strategies-panel');
  if (strat) strat.innerHTML = '';
  var recap = document.getElementById('gen33-pipeline-recap');
  if (recap) recap.innerHTML = '';
  var q = document.getElementById('gen33-question');
  if (q) q.innerHTML = '';

  // Reset sliderBound so strategies rebuild properly
  generationState.sliderBound = false;

  // Rebuild initial context
  buildGen33Context();

  var takeaway = document.getElementById('gen33-takeaway');
  if (takeaway) takeaway.innerHTML = GEN33_TAKEAWAYS[0];
}
