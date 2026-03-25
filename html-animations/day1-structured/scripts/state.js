const state = {
  nav: {
    order: SLIDE_ORDER.slice(),
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
  registry: {
    order: [],
    byId: {}
  },
  timers: {
    timeouts: []
  },
  inference: {
    step: 0,
    pendingBlank: false,
    blankTimer: null
  },
  training: {
    phase: 0
  },
  completion: {
    fixesRevealed: {
      chat: false,
      math: false,
      code: false
    },
    pendingBridge: false,
    bridgeTimer: null
  }
};

const inferenceState = state.inference;
const trainingState = state.training;
const completionState = state.completion;

function removeTrackedTimeout(timerId) {
  state.timers.timeouts = state.timers.timeouts.filter((id) => id !== timerId);
}

function clearTrackedTimeout(timerId) {
  if (timerId === null || timerId === undefined) return;
  window.clearTimeout(timerId);
  removeTrackedTimeout(timerId);
}

function setTrackedTimeout(callback, delay) {
  const timerId = window.setTimeout(function() {
    removeTrackedTimeout(timerId);
    callback();
  }, delay);
  state.timers.timeouts.push(timerId);
  return timerId;
}

function clearTrackedTimeouts() {
  state.timers.timeouts.forEach((timerId) => {
    window.clearTimeout(timerId);
  });
  state.timers.timeouts = [];
}
