/**
 * Iris Documentation - Simplified Layout Manager
 * Minimal JavaScript for enhanced UX without breaking RTD functionality
 */

(function() {
  'use strict';

  // Simple debounce utility
  function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
      const later = () => {
        clearTimeout(timeout);
        func(...args);
      };
      clearTimeout(timeout);
      timeout = setTimeout(later, wait);
    };
  }

  // Enhance search input with better UX
  function enhanceSearchInput() {
    const searchInput = document.querySelector('.wy-side-nav-search input[type="text"]');
    if (!searchInput) return;

    // Add placeholder text if not already set
    if (!searchInput.placeholder) {
      searchInput.placeholder = 'Search documentation...';
    }

    // Add smooth focus/blur animations
    searchInput.addEventListener('focus', function() {
      this.parentElement.classList.add('search-focused');
    });

    searchInput.addEventListener('blur', function() {
      this.parentElement.classList.remove('search-focused');
    });
  }

  // Add smooth scroll behavior for navigation links
  function enhanceNavigation() {
    const navLinks = document.querySelectorAll('.wy-menu-vertical a[href^="#"]');

    navLinks.forEach(link => {
      link.addEventListener('click', function(e) {
        const href = this.getAttribute('href');
        const target = document.querySelector(href);

        if (target) {
          e.preventDefault();
          target.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
          });

          // Update URL without jumping
          history.pushState(null, null, href);
        }
      });
    });
  }

  // Add copy button functionality for code blocks (if sphinx-copybutton is not available)
  function addCopyButtons() {
    // Only add if sphinx-copybutton is not already present
    if (document.querySelector('.copybtn')) return;

    const codeBlocks = document.querySelectorAll('.highlight pre');

    codeBlocks.forEach(block => {
      const button = document.createElement('button');
      button.className = 'copy-btn';
      button.innerHTML = '📋';
      button.title = 'Copy to clipboard';
      button.style.cssText = `
        position: absolute;
        top: 8px;
        right: 8px;
        background: var(--primary-600);
        color: white;
        border: none;
        border-radius: 4px;
        padding: 4px 8px;
        font-size: 12px;
        cursor: pointer;
        opacity: 0.7;
        transition: opacity 0.2s;
      `;

      button.addEventListener('click', async function() {
        try {
          await navigator.clipboard.writeText(block.textContent);
          button.innerHTML = '✅';
          button.title = 'Copied!';
          setTimeout(() => {
            button.innerHTML = '📋';
            button.title = 'Copy to clipboard';
          }, 2000);
        } catch (err) {
          console.warn('Could not copy text: ', err);
        }
      });

      button.addEventListener('mouseenter', function() {
        this.style.opacity = '1';
      });

      button.addEventListener('mouseleave', function() {
        this.style.opacity = '0.7';
      });

      // Add button to code block container
      const container = block.parentElement;
      container.style.position = 'relative';
      container.appendChild(button);
    });
  }

  // Add keyboard navigation enhancement
  function enhanceKeyboardNavigation() {
    document.addEventListener('keydown', function(e) {
      // Alt + S to focus search
      if (e.altKey && e.key === 's') {
        e.preventDefault();
        const searchInput = document.querySelector('.wy-side-nav-search input[type="text"]');
        if (searchInput) {
          searchInput.focus();
          searchInput.select();
        }
      }

      // Escape to blur search
      if (e.key === 'Escape') {
        const searchInput = document.querySelector('.wy-side-nav-search input[type="text"]:focus');
        if (searchInput) {
          searchInput.blur();
        }
      }
    });
  }

  // Add loading state for better perceived performance
  function addLoadingStates() {
    // Add loading class to body initially
    document.body.classList.add('loading');

    // Remove loading class when everything is ready
    window.addEventListener('load', function() {
      setTimeout(() => {
        document.body.classList.remove('loading');
        document.body.classList.add('loaded');
      }, 100);
    });
  }

  // Add pan/zoom functionality to Mermaid diagrams and scale to fit
  function setupMermaidInteractive() {
    // Wait for Mermaid to fully render
    setTimeout(() => {
      const mermaidContainers = document.querySelectorAll('.mermaid');
      
      mermaidContainers.forEach(container => {
        const svg = container.querySelector('svg');
        if (!svg) return;
        
        try {
          // Get dimensions
          const containerWidth = container.clientWidth - 32;
          const svgBBox = svg.getBBox();
          const svgWidth = svgBBox.width;
          const svgHeight = svgBBox.height;
          
          // Set viewBox to ensure full diagram is visible
          svg.setAttribute('viewBox', `${svgBBox.x} ${svgBBox.y} ${svgWidth} ${svgHeight}`);
          svg.removeAttribute('width');
          svg.removeAttribute('height');
          
          // Calculate initial scale to fit perfectly
          let initialScale = 1;
          if (svgWidth > containerWidth) {
            initialScale = containerWidth / svgWidth; // Exact fit
          }
          
          // Center the diagram
          const scaledWidth = svgWidth * initialScale;
          const centerOffset = (containerWidth - scaledWidth) / 2;
          
          // Setup pan/zoom state
          let isPanning = false;
          let startPoint = { x: 0, y: 0 };
          let currentTranslate = { x: centerOffset, y: 0 };
          let currentScale = initialScale;
          
          // Apply initial transform
          const updateTransform = () => {
            svg.style.transform = `translate(${currentTranslate.x}px, ${currentTranslate.y}px) scale(${currentScale})`;
            svg.style.transformOrigin = '0 0';
          };
          updateTransform();
          
          // Mouse/touch event handlers for panning
          const startPan = (e) => {
            isPanning = true;
            const point = e.touches ? e.touches[0] : e;
            startPoint = {
              x: point.clientX - currentTranslate.x,
              y: point.clientY - currentTranslate.y
            };
            container.style.cursor = 'grabbing';
            e.preventDefault();
          };
          
          const doPan = (e) => {
            if (!isPanning) return;
            const point = e.touches ? e.touches[0] : e;
            currentTranslate = {
              x: point.clientX - startPoint.x,
              y: point.clientY - startPoint.y
            };
            updateTransform();
            e.preventDefault();
          };
          
          const endPan = () => {
            isPanning = false;
            container.style.cursor = 'grab';
          };
          
          // Zoom with mouse wheel
          const doZoom = (e) => {
            e.preventDefault();
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            const newScale = currentScale * delta;
            
            // Limit zoom range
            if (newScale >= 0.1 && newScale <= 5) {
              // Get mouse position relative to container
              const rect = container.getBoundingClientRect();
              const mouseX = e.clientX - rect.left;
              const mouseY = e.clientY - rect.top;
              
              // Adjust translate to zoom around mouse position
              currentTranslate.x = mouseX - (mouseX - currentTranslate.x) * delta;
              currentTranslate.y = mouseY - (mouseY - currentTranslate.y) * delta;
              currentScale = newScale;
              
              updateTransform();
            }
          };
          
          // Add event listeners
          container.style.cursor = 'grab';
          container.addEventListener('mousedown', startPan);
          container.addEventListener('mousemove', doPan);
          container.addEventListener('mouseup', endPan);
          container.addEventListener('mouseleave', endPan);
          container.addEventListener('touchstart', startPan);
          container.addEventListener('touchmove', doPan);
          container.addEventListener('touchend', endPan);
          container.addEventListener('wheel', doZoom, { passive: false });
          
          // Double-click to reset
          container.addEventListener('dblclick', () => {
            currentScale = initialScale;
            currentTranslate = { x: centerOffset, y: 0 };
            updateTransform();
          });
          
        } catch (e) {
          console.warn('Could not setup Mermaid interactive:', e);
        }
      });
    }, 1000);
  }

  // Main initialization function
  function init() {
    try {
      enhanceSearchInput();
      enhanceNavigation();
      addCopyButtons();
      enhanceKeyboardNavigation();
      addLoadingStates();
      setupMermaidInteractive();
    } catch (error) {
      console.warn('Iris Layout Manager: Some enhancements failed to initialize:', error);
    }
  }

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  // Handle page changes for single-page applications
  window.addEventListener('popstate', debounce(init, 100));

  // Export for debugging (optional)
  if (typeof window !== 'undefined') {
    window.IrisLayoutManager = {
      init,
      enhanceSearchInput,
      enhanceNavigation,
      addCopyButtons
    };
  }

})();
