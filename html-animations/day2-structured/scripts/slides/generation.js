/* Slide 33 — You Now Know the Transformer */

var GEN33_MAX_STEP = 1;

var GEN33_CLASSES = ['gen33-show-insights'];

var GEN33_TAKEAWAYS = [
  'Every LLM runs this same pipeline. You now know every box.',
  'Tokenize \u2192 Represent \u2192 Understand \u00d7 L \u2192 Predict. One token at a time.'
];

var GEN33_PIPELINE_SEGS = [
  { text: 'tokens',     cls: 'gen33-seg-embed' },
  { text: '\u2192',     cls: 'gen33-seg-arrow' },
  { text: 'embed',      cls: 'gen33-seg-embed' },
  { text: '+',          cls: 'gen33-seg-add' },
  { text: 'PE',         cls: 'gen33-seg-position' },
  { text: '\u2192',     cls: 'gen33-seg-arrow' },
  { text: '[',          cls: 'gen33-seg-bracket' },
  { text: 'LN',         cls: 'gen33-seg-norm' },
  { text: '\u2192',     cls: 'gen33-seg-arrow-sm' },
  { text: 'Attn',       cls: 'gen33-seg-attn' },
  { text: '\u2192',     cls: 'gen33-seg-arrow-sm' },
  { text: '+',          cls: 'gen33-seg-add' },
  { text: '\u2192',     cls: 'gen33-seg-arrow-sm' },
  { text: 'LN',         cls: 'gen33-seg-norm' },
  { text: '\u2192',     cls: 'gen33-seg-arrow-sm' },
  { text: 'FFN',        cls: 'gen33-seg-ffn' },
  { text: '\u2192',     cls: 'gen33-seg-arrow-sm' },
  { text: '+',          cls: 'gen33-seg-add' },
  { text: '] \u00d7 L', cls: 'gen33-seg-bracket' },
  { text: '\u2192',     cls: 'gen33-seg-arrow' },
  { text: 'LM head',    cls: 'gen33-seg-head' },
  { text: '\u2192',     cls: 'gen33-seg-arrow' },
  { text: 'next token', cls: 'gen33-seg-decode' }
];

var GEN33_INSIGHTS = [
  {
    title: 'Tokenize',
    body: 'The model reads subword chunks, not characters or words. Vocabulary boundaries shape what it can and can\'t reason about.',
    color: 'blue'
  },
  {
    title: 'Attend',
    body: 'Each token attends to all others. Attention weights are learned, contextual, and computed across multiple heads simultaneously.',
    color: 'purple'
  },
  {
    title: 'Deepen',
    body: 'L stacked blocks refine meaning layer by layer. Neither attention nor FFN changes the sequence shape \u2014 just what\'s in each row.',
    color: 'cyan'
  },
  {
    title: 'Predict',
    body: 'Every output is a probability over the vocabulary. Generation is next-token prediction, repeated one token at a time.',
    color: 'green'
  }
];

function buildGen33Pipeline() {
  var pipeline = document.getElementById('close33-pipeline');
  if (!pipeline || pipeline.children.length > 0) return;
  GEN33_PIPELINE_SEGS.forEach(function(seg) {
    pipeline.appendChild(createEl('span', {
      className: 'gen33-seg ' + seg.cls,
      text: seg.text
    }));
  });
}

function buildGen33Insights() {
  var grid = document.getElementById('close33-insights');
  if (!grid || grid.children.length > 0) return;
  GEN33_INSIGHTS.forEach(function(ins) {
    var card = createEl('div', {
      className: 'close33-insight-card close33-insight-' + ins.color
    });
    card.appendChild(createEl('div', {
      className: 'close33-insight-title',
      text: ins.title
    }));
    card.appendChild(createEl('div', {
      className: 'close33-insight-body',
      text: ins.body
    }));
    grid.appendChild(card);
  });
}

function setGenerationStep(step) {
  var slide = document.getElementById('slide-33');
  var takeaway = document.getElementById('close33-takeaway');
  if (!slide || !takeaway) return;

  var clamped = Math.max(0, Math.min(GEN33_MAX_STEP, step));
  generationState.step = clamped;

  GEN33_CLASSES.forEach(function(cls, idx) {
    slide.classList.toggle(cls, clamped >= idx + 1);
  });

  if (clamped >= 1) {
    buildGen33Insights();
  }

  takeaway.innerHTML = GEN33_TAKEAWAYS[clamped] || GEN33_TAKEAWAYS[0];
}

function initGenerationSlide() {
  if (!generationState.initialized) {
    buildGen33Pipeline();
    generationState.initialized = true;
  }
  setGenerationStep(generationState.step || 0);
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
  var takeaway = document.getElementById('close33-takeaway');
  if (takeaway) takeaway.innerHTML = GEN33_TAKEAWAYS[0];
}
