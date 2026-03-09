const state = {
  nav: {
    slides: [],
    total: 0,
    current: 0
  },
  ui: {
    btnPrev: null,
    btnNext: null,
    btnSkip: null,
    slideCounter: null,
    progressFill: null
  },
  projection: {
    initialized: false,
    activeLens: DEFAULT_PROJECTION_LENS,
    currentPositions: null,
    rafId: null,
    animStart: 0,
    animFrom: null,
    animTo: null,
    canvas: null,
    ctx: null,
    resizeBound: false,
    readoutTimer: null
  },
  attentionIntro: {
    initialized: false,
    step: 0,
    resizeBound: false,
    overlayTimer: null,
    flowTimers: []
  },
  attentionQkv: {
    initialized: false,
    step: 0,
    resizeBound: false,
    overlayTimer: null,
    compareTimers: [],
    compareDone: false,
    compareVisibleCount: 0
  },
  registry: {
    order: [],
    byId: {}
  },
  dev: {
    listenerCounts: {},
    listenerKeys: new Set(),
    snapshots: []
  }
};

const projectionState = state.projection;
const attentionIntroState = state.attentionIntro;
const attentionQkvState = state.attentionQkv;

