const FIT_TARGET_WIDTH = 1366;
const FIT_TARGET_HEIGHT = 768;
const FIT_SCALE_EPSILON = 0.001;

const layoutState = {
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
  const shell = createEl('div', { className: 'slide-fit-shell' });
  var children = Array.from(slideEl.childNodes);
  for (var i = 0; i < children.length; i++) {
    var child = children[i];
    // Keep section-tags outside the shell — they sit in the slide padding
    if (child.nodeType === 1 && child.classList.contains('section-tag')) continue;
    shell.appendChild(child);
  }
  slideEl.appendChild(shell);
}

function initializeSlideFitSystem() {
  $$('.slide').forEach((slideEl) => {
    wrapSlideContent(slideEl);
  });

  if (layoutState.resizeBound) return;
  addTrackedListener(window, 'resize', () => {
    if (layoutState.syntheticResizeGuard) return;
    scheduleActiveSlideFit({ reason: 'window-resize', dispatchResize: false });
  });
  layoutState.resizeBound = true;
}

function getSlideAvailableBox(slideEl) {
  if (!slideEl) return null;
  const styles = window.getComputedStyle(slideEl);
  const paddingX = parseFloat(styles.paddingLeft) + parseFloat(styles.paddingRight);
  const paddingY = parseFloat(styles.paddingTop) + parseFloat(styles.paddingBottom);
  return {
    width: Math.max(0, slideEl.clientWidth - paddingX),
    height: Math.max(0, slideEl.clientHeight - paddingY)
  };
}

function isTargetViewport() {
  return Math.abs(window.innerWidth - FIT_TARGET_WIDTH) <= 8 && Math.abs(window.innerHeight - FIT_TARGET_HEIGHT) <= 8;
}

function logFitWarning(slideEl, metrics) {
  if (!DEV || !slideEl || !isTargetViewport()) return;
  if (metrics.naturalHeight <= metrics.availableHeight + 1 && metrics.naturalWidth <= metrics.availableWidth + 1 && metrics.scale >= 0.9) return;

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
  requestAnimationFrame(() => {
    layoutState.syntheticResizeGuard = false;
  });
}

function fitSlide(slideEl, options) {
  const shell = getSlideFitShell(slideEl);
  if (!slideEl || !shell) return 1;

  const opts = Object.assign({ dispatchResize: true }, options || {});
  const available = getSlideAvailableBox(slideEl);
  if (!available) return 1;
  const prevScale = parseFloat(slideEl.style.getPropertyValue('--slide-fit-scale') || '1');

  shell.style.transform = 'scale(1)';
  slideEl.style.setProperty('--slide-fit-scale', '1');

  // Temporarily unset height so flex children expand to natural size
  // instead of shrinking to fit, giving us the true content height.
  shell.style.height = 'auto';
  const naturalWidth = Math.max(shell.scrollWidth, shell.offsetWidth, shell.getBoundingClientRect().width);
  const naturalHeight = Math.max(shell.scrollHeight, shell.offsetHeight, shell.getBoundingClientRect().height);
  const widthRatio = available.width > 0 ? available.width / naturalWidth : 1;
  const heightRatio = available.height > 0 ? available.height / naturalHeight : 1;
  const fittedScale = Math.min(1, widthRatio, heightRatio);
  const scale = Number.isFinite(fittedScale) && fittedScale > 0 ? fittedScale : 1;

  // Set shell layout height to available/scale so internal flex layout has
  // enough room for all content. The transform then shrinks it to fit visually.
  shell.style.height = (available.height / scale) + 'px';
  shell.style.transform = 'scale(' + scale + ')';
  slideEl.style.setProperty('--slide-fit-scale', scale.toFixed(4));
  layoutState.currentScale = scale;

  logFitWarning(slideEl, {
    naturalHeight,
    naturalWidth,
    availableHeight: available.height,
    availableWidth: available.width,
    scale
  });

  if (opts.dispatchResize && Math.abs(scale - prevScale) > FIT_SCALE_EPSILON) {
    dispatchSyntheticResizeOnce();
  }

  return scale;
}

function scheduleActiveSlideFit(options) {
  const opts = Object.assign({ dispatchResize: true }, options || {});
  layoutState.fitSeq += 1;
  const fitSeq = layoutState.fitSeq;
  if (layoutState.pendingRaf) {
    cancelAnimationFrame(layoutState.pendingRaf);
  }

  layoutState.pendingRaf = requestAnimationFrame(() => {
    const activeSlide = state.nav.slides[state.nav.current] || document.querySelector('.slide.active');
    if (!activeSlide) return;
    typesetMath(activeSlide).then(() => {
      requestAnimationFrame(() => {
        if (fitSeq !== layoutState.fitSeq) return;
        fitSlide(activeSlide, opts);
      });
    });
  });
}
