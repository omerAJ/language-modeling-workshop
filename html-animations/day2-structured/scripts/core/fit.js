const DECK_WIDTH = 1920;
const DECK_HEIGHT = 1080;

const layoutState = {
  resizeBound: false,
  syntheticResizeGuard: false,
  currentScale: 1,
  refreshSeq: 0,
  pendingRaf: null
};

function getDeckRoot() {
  return document.getElementById('deckRoot');
}

function getDeckViewport() {
  return document.getElementById('deckViewport');
}

function applyDeckScale() {
  const deckRoot = getDeckRoot();
  const deckViewport = getDeckViewport();
  if (!deckRoot || !deckViewport) return 1;

  const scale = Math.min(
    deckViewport.clientWidth / DECK_WIDTH,
    deckViewport.clientHeight / DECK_HEIGHT
  );
  const nextScale = Number.isFinite(scale) && scale > 0 ? scale : 1;
  deckRoot.style.transform = 'translate(-50%, -50%) scale(' + nextScale + ')';
  deckRoot.style.setProperty('--deck-scale', nextScale.toFixed(4));
  layoutState.currentScale = nextScale;
  return nextScale;
}

function initializeDeckScaleSystem() {
  applyDeckScale();

  if (layoutState.resizeBound) return;
  addTrackedListener(window, 'resize', () => {
    if (layoutState.syntheticResizeGuard) return;
    scheduleDeckRefresh({ reason: 'window-resize', dispatchResize: false, typeset: false });
  });
  layoutState.resizeBound = true;
}

function dispatchSyntheticResizeOnce() {
  if (layoutState.syntheticResizeGuard) return;
  layoutState.syntheticResizeGuard = true;
  window.dispatchEvent(new Event('resize'));
  requestAnimationFrame(() => {
    layoutState.syntheticResizeGuard = false;
  });
}

function scheduleDeckRefresh(options) {
  const opts = Object.assign({ dispatchResize: true, typeset: true }, options || {});
  layoutState.refreshSeq += 1;
  const refreshSeq = layoutState.refreshSeq;
  if (layoutState.pendingRaf) {
    cancelAnimationFrame(layoutState.pendingRaf);
  }

  layoutState.pendingRaf = requestAnimationFrame(() => {
    const activeSlide = state.nav.slides[state.nav.current] || document.querySelector('.slide.active');

    const finalize = () => {
      requestAnimationFrame(() => {
        if (refreshSeq !== layoutState.refreshSeq) return;
        applyDeckScale();
        if (opts.dispatchResize) {
          dispatchSyntheticResizeOnce();
        }
      });
    };

    if (!opts.typeset || !activeSlide) {
      finalize();
      return;
    }

    typesetMath(activeSlide).then(finalize);
  });
}
