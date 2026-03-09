const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

const DEV = false;

const LENS_LABELS = {
  pet: 'Pet / companionship',
  feline: 'Feline family',
  care: 'Care context'
};

const DEFAULT_PROJECTION_LENS = 'pet';

const LENS_MEANINGS = {
  pet: 'pet and companionship features',
  feline: 'feline-family features',
  care: 'care and ownership features'
};

const LENS_STATES = {
  pet: {
    cat:[0.30,0.55],
    dog:[0.22,0.60],
    hamster:[0.20,0.45],
    vet:[0.35,0.70],
    litter:[0.48,0.30],
    kibble:[0.50,0.45],
    tiger:[0.80,0.60],
    lion:[0.82,0.45],
    leopard:[0.75,0.50]
  },
  feline: {
    cat:[0.55,0.55],
    tiger:[0.70,0.65],
    lion:[0.72,0.45],
    leopard:[0.68,0.55],
    dog:[0.20,0.65],
    hamster:[0.18,0.40],
    vet:[0.40,0.30],
    litter:[0.48,0.12],
    kibble:[0.55,0.20]
  },
  care: {
    cat:[0.45,0.55],
    litter:[0.35,0.45],
    kibble:[0.52,0.45],
    vet:[0.48,0.70],
    dog:[0.30,0.65],
    hamster:[0.32,0.30],
    tiger:[0.80,0.70],
    lion:[0.82,0.45],
    leopard:[0.75,0.55]
  }
};

const LENS_NEIGHBORS = {
  pet: ['dog', 'hamster', 'vet'],
  feline: ['leopard', 'tiger', 'lion'],
  care: ['litter', 'kibble', 'vet']
};

const LENS_EMPHASIS = {
  pet: ['cat', 'dog', 'hamster', 'vet'],
  feline: ['cat', 'tiger', 'lion', 'leopard'],
  care: ['cat', 'litter', 'kibble', 'vet']
};

const TOKEN_GROUPS = {
  cat: 'anchor',
  dog: 'pets',
  hamster: 'pets',
  tiger: 'bigcats',
  lion: 'bigcats',
  leopard: 'bigcats',
  litter: 'care',
  kibble: 'care',
  vet: 'care'
};

const TOKEN_LABEL_OFFSETS = {
  dog: [9, -12],
  hamster: [9, 12],
  tiger: [9, -12],
  lion: [9, 12],
  leopard: [9, -14],
  litter: [9, 12],
  kibble: [9, 12],
  vet: [9, -12]
};

const PROJECTION_TOKENS = Object.keys(LENS_STATES[DEFAULT_PROJECTION_LENS]);
const ATTN_INTRO_TOKENS = ['cat', 'sat', 'on', 'the', 'mat'];
const ATTN_INTRO_FLOW_SOURCES = ['cat', 'on', 'the', 'mat', 'sat'];
const ATTN_INTRO_FOCUS = 'sat';
const ATTN_INTRO_FOOTER = 'Next: how does sat decide which tokens matter — and what to copy from them?';
const ATTN_INTRO_FLOW_HEIGHTS = {
  cat: 0.08,
  on: 0.12,
  the: 0.16,
  mat: 0.2,
  sat: 0.18
};
const ATTN_INTRO_FLOW_ANIM_MS = 380;
const ATTN_INTRO_FLOW_HEAD_FADE_MS = 150;
const ATTN_INTRO_MAX_STEP = 8;
const ATTN_QKV_TOKENS = ['cat', 'sat', 'on', 'the', 'mat'];
const ATTN_QKV_FOCUS = 'sat';
const ATTN_QKV_COMPARE_TOKENS = ['cat', 'sat', 'on', 'the', 'mat'];
const ATTN_QKV_COMPARE_CURVE_FACTORS = {
  cat: 0.4,
  sat: 0.52,
  on: 0.58,
  the: 0.64,
  mat: 0.46
};
const ATTN_QKV_SCORE_LEVELS = {
  cat: 1,
  sat: 0.44,
  on: 0.58,
  the: 0.34,
  mat: 0.52
};
const ATTN_QKV_TAKEAWAYS = [
  'We start from embeddings x₁…xₙ',
  'Project each x into a key: k = x W_K',
  'Also project into a value: v = x W_V (this is what gets copied)',
  'For the focus token, build a query: q_sat = x_sat W_Q',
  'Compute scores: score_j = similarity(q_sat, k_j)'
];
const ATTN_QKV_COMPARE_DRAW_MS = 360;
const ATTN_QKV_COMPARE_HEAD_FADE_MS = 140;
const ATTN_QKV_COMPARE_STAGGER_MS = 190;
const ATTN_QKV_SCORE_REVEAL_DELAY_MS = 140;
const ATTN_QKV_MAX_STEP = 4;

