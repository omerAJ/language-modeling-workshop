const state = {
  nav: {
    order: SLIDE_ORDER.slice(),
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
  registry: {
    order: [],
    byId: {}
  },
  inference: {
    step: 0
  },
  training: {
    phase: 0
  },
  completion: {
    fixesRevealed: {
      chat: false,
      math: false,
      code: false
    }
  }
};

const inferenceState = state.inference;
const trainingState = state.training;
const completionState = state.completion;
