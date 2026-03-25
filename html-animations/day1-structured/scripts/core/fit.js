var DEV = false;
var FIT_TARGET_WIDTH = 1366;
var FIT_TARGET_HEIGHT = 768;
var FIT_SCALE_EPSILON = 0.001;

var layoutState = {
  resizeBound: false,
  syntheticResizeGuard: false,
  currentScale: 1,
  fitSeq: 0,
  pendingRaf: null
};

function getSlideFitShell(slideEl) {
  if (!slideEl) return null;
  return slideEl.querySelector(':scope > .slide-fit-shell');
}

function wrapSlideContent(slideEl) {
  if (!slideEl || getSlideFitShell(slideEl)) return;
  var shell = document.createElement('div');
  shell.className = 'slide-fit-shell';
  var children = Array.from(slideEl.childNodes);
  for (var i = 0; i < children.length; i++) {
    var child = children[i];
    if (child.nodeType === 1 && child.classList.contains('section-tag')) continue;
    shell.appendChild(child);
  }
  slideEl.appendChild(shell);
}

function initializeSlideFitSystem() {
  $$('.slide').forEach(function(slideEl) {
    wrapSlideContent(slideEl);
  });

  if (layoutState.resizeBound) return;
  window.addEventListener('resize', function() {
    if (layoutState.syntheticResizeGuard) return;
    scheduleActiveSlideFit({ reason: 'window-resize', dispatchResize: false });
  });
  layoutState.resizeBound = true;
}

function getSlideAvailableBox(slideEl) {
  if (!slideEl) return null;
  var styles = window.getComputedStyle(slideEl);
  var paddingX = parseFloat(styles.paddingLeft) + parseFloat(styles.paddingRight);
  var paddingY = parseFloat(styles.paddingTop) + parseFloat(styles.paddingBottom);
  return {
    width: Math.max(0, slideEl.clientWidth - paddingX),
    height: Math.max(0, slideEl.clientHeight - paddingY)
  };
}

function isTargetViewport() {
  return Math.abs(window.innerWidth - FIT_TARGET_WIDTH) <= 8 &&
         Math.abs(window.innerHeight - FIT_TARGET_HEIGHT) <= 8;
}

function logFitWarning(slideEl, metrics) {
  if (!DEV || !slideEl || !isTargetViewport()) return;
  if (metrics.naturalHeight <= metrics.availableHeight + 1 &&
      metrics.naturalWidth <= metrics.availableWidth + 1 &&
      metrics.scale >= 0.9) return;

  console.warn('[DEV][fit]', {
    slideId: slideEl.id,
    naturalHeight: Math.round(metrics.naturalHeight),
    naturalWidth: Math.round(metrics.naturalWidth),
    availableHeight: Math.round(metrics.availableHeight),
    availableWidth: Math.round(metrics.availableWidth),
    scale: Number(metrics.scale.toFixed(3))
  });
}

function dispatchSyntheticResizeOnce() {
  if (layoutState.syntheticResizeGuard) return;
  layoutState.syntheticResizeGuard = true;
  window.dispatchEvent(new Event('resize'));
  requestAnimationFrame(function() {
    layoutState.syntheticResizeGuard = false;
  });
}

function fitSlide(slideEl, options) {
  var shell = getSlideFitShell(slideEl);
  if (!slideEl || !shell) return 1;

  var opts = Object.assign({ dispatchResize: true }, options || {});
  var available = getSlideAvailableBox(slideEl);
  if (!available) return 1;
  var prevScale = parseFloat(slideEl.style.getPropertyValue('--slide-fit-scale') || '1');

  shell.style.transform = 'scale(1)';
  slideEl.style.setProperty('--slide-fit-scale', '1');

  shell.style.height = 'auto';
  var naturalWidth = Math.max(shell.scrollWidth, shell.offsetWidth, shell.getBoundingClientRect().width);
  var naturalHeight = Math.max(shell.scrollHeight, shell.offsetHeight, shell.getBoundingClientRect().height);
  var widthRatio = available.width > 0 ? available.width / naturalWidth : 1;
  var heightRatio = available.height > 0 ? available.height / naturalHeight : 1;
  var fittedScale = Math.min(1, widthRatio, heightRatio);
  var scale = Number.isFinite(fittedScale) && fittedScale > 0 ? fittedScale : 1;

  shell.style.height = (available.height / scale) + 'px';
  shell.style.transform = 'scale(' + scale + ')';
  slideEl.style.setProperty('--slide-fit-scale', scale.toFixed(4));
  layoutState.currentScale = scale;

  logFitWarning(slideEl, {
    naturalHeight: naturalHeight,
    naturalWidth: naturalWidth,
    availableHeight: available.height,
    availableWidth: available.width,
    scale: scale
  });

  if (opts.dispatchResize && Math.abs(scale - prevScale) > FIT_SCALE_EPSILON) {
    dispatchSyntheticResizeOnce();
  }

  return scale;
}

function scheduleActiveSlideFit(options) {
  var opts = Object.assign({ dispatchResize: true }, options || {});
  layoutState.fitSeq += 1;
  var fitSeq = layoutState.fitSeq;
  if (layoutState.pendingRaf) {
    cancelAnimationFrame(layoutState.pendingRaf);
  }

  layoutState.pendingRaf = requestAnimationFrame(function() {
    var activeSlide = state.nav.slides[state.nav.current] || document.querySelector('.slide.active');
    if (!activeSlide) return;
    typesetMath(activeSlide).then(function() {
      requestAnimationFrame(function() {
        if (fitSeq !== layoutState.fitSeq) return;
        fitSlide(activeSlide, opts);
      });
    });
  });
}
