const state = {
  nav: {
    slides: [],
    total: 0,
    current: 0,
    history: []
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
    overlayTimer: null,
    flowTimers: [],
    resizeBound: false
  },
  attentionP1: {
    initialized: false,
    step: 0,
    overlayTimer: null,
    resizeBound: false
  },
  attentionQkv: {
    initialized: false,
    step: 0,
    overlayTimer: null,
    compareTimers: [],
    compareDone: false,
    compareVisibleCount: 0,
    resizeBound: false
  },
  attentionWeights: {
    initialized: false,
    step: 0
  },
  attentionStep4: {
    initialized: false,
    step: 0,
    overlayTimer: null,
    compareTimers: [],
    compareDone: false,
    compareVisibleCount: 0,
    pairTimers: [],
    pairRafIds: [],
    firstPairDone: false,
    pairingDone: false,
    pairVisibleCount: 0,
    aggTimers: [],
    aggDone: false,
    aggTermsVisibleCount: 0,
    aggCollapsed: false,
    mergeTimers: [],
    mergeRafIds: [],
    mergeDone: false,
    mergeVisibleCount: 0,
    residualTimers: [],
    residualRafIds: [],
    residualDone: false,
    resizeBound: false
  },
  attentionMatrix: {
    initialized: false,
    step: 0,
    timers: [],
    rafIds: [],
    tokenMatrixDone: false,
    xMatrixDone: false,
    tokenVisibleCount: 0,
    xVisibleCount: 0,
    projectionDone: false,
    projectionVisible: false,
    projectionTimers: [],
    projectionRafIds: [],
    scoreCenteredDone: false,
    scoreTransposedDone: false,
    scoreMatrixDone: false,
    scoreVisible: false,
    scoreVisibleCount: 0,
    maskProblemDone: false,
    maskAppliedDone: false,
    maskQuestionDone: false,
    maskVisibleCount: 0,
    maskMode: 'none',
    postScoreCenteredDone: false,
    scaledMatrixDone: false,
    attentionMatrixDone: false,
    valueMatrixVisibleDone: false,
    outputMatrixDone: false,
    postScoreMode: 'none',
    postScoreVisibleCount: 0,
    outputVisibleCount: 0,
    scoreTimers: [],
    scoreRafIds: [],
    resizeBound: false
  },
  attentionMultiHead: {
    initialized: false,
    step: 0,
    timers: [],
    rafIds: [],
    splitDone: false,
    projDone: false,
    attnDone: false,
    outputDone: false,
    outputVisibleCount: 0,
    concatDone: false,
    outputProjectionDone: false,
    combineVisible: false,
    resizeBound: false
  },
  attentionPosition: {
    initialized: false,
    step: 0,
    timers: [],
    rafIds: [],
    resizeBound: false
  },
  orderProblem: {
    initialized: false
  },
  positionSignal: {
    initialized: false
  },
  ffnCombined: {
    initialized: false
  },
  gptBlock: {
    initialized: false,
    step: 0
  },
  outputHead: {
    initialized: false,
    step: 0,
    timers: []
  },
  generation: {
    initialized: false,
    step: 0,
    timers: [],
    sliderBound: false
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
const attentionP1State = state.attentionP1;
const attentionQkvState = state.attentionQkv;
const attentionWeightsState = state.attentionWeights;
const attentionStep4State = state.attentionStep4;
const attentionMatrixState = state.attentionMatrix;
const attentionMultiHeadState = state.attentionMultiHead;
const attentionPositionState = state.attentionPosition;
const orderProblemState = state.orderProblem;
const positionSignalState = state.positionSignal;
const ffnCombinedState = state.ffnCombined;
const gptBlockState = state.gptBlock;
const outputHeadState = state.outputHead;
const generationState = state.generation;
