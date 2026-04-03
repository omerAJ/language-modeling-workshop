var DECK_WIDTH = 1920;
var DECK_HEIGHT = 1080;

var layoutState = {
  resizeBound: false,
  syntheticResizeGuard: false,
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
  var deckRoot = getDeckRoot();
  var deckViewport = getDeckViewport();
  if (!deckRoot || !deckViewport) return 1;

  var scale = Math.min(
    deckViewport.clientWidth / DECK_WIDTH,
    deckViewport.clientHeight / DECK_HEIGHT
  );
  var nextScale = Number.isFinite(scale) && scale > 0 ? scale : 1;
  deckRoot.style.transform = 'translate(-50%, -50%) scale(' + nextScale + ')';
  deckRoot.style.setProperty('--deck-scale', nextScale.toFixed(4));
  return nextScale;
}

function dispatchSyntheticResizeOnce() {
  if (layoutState.syntheticResizeGuard) return;
  layoutState.syntheticResizeGuard = true;
  window.dispatchEvent(new Event('resize'));
  requestAnimationFrame(function() {
    layoutState.syntheticResizeGuard = false;
  });
}

function initializeDeckScaleSystem() {
  applyDeckScale();

  if (layoutState.resizeBound) return;
  window.addEventListener('resize', function() {
    if (layoutState.syntheticResizeGuard) return;
    scheduleDeckRefresh({ reason: 'window-resize', dispatchResize: false, typeset: false });
  });
  layoutState.resizeBound = true;
}

function scheduleDeckRefresh(options) {
  var opts = Object.assign({ dispatchResize: true, typeset: true }, options || {});
  layoutState.refreshSeq += 1;
  var refreshSeq = layoutState.refreshSeq;
  if (layoutState.pendingRaf) {
    cancelAnimationFrame(layoutState.pendingRaf);
  }

  layoutState.pendingRaf = requestAnimationFrame(function() {
    var activeSlide = state.nav.slides[state.nav.current] || document.querySelector('.slide.active');

    var finalize = function() {
      requestAnimationFrame(function() {
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

function initializeSlideFitSystem() {
  initializeDeckScaleSystem();
}

function scheduleActiveSlideFit(options) {
  scheduleDeckRefresh(options);
}
