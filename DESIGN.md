---
name: VeriFact AI
description: Autonomous multi-agent claim verification pipeline with Midnight Transit Diagram visual identity
colors:
  midnight: "#0F1B3D"
  surface: "#16244D"
  porcelain: "#F4F1EA"
  line-factcheck: "#B9A2F2"
  line-crossref: "#2BB3A3"
  line-linguistic: "#F4F1EA"
  verdict-false: "#DC2626"
  verdict-mixed: "#F59E0B"
  verdict-true: "#10B981"
  verdict-none: "#8A94B0"
  bias-left: "#3B6FD8"
  bias-leanleft: "#8FB0F0"
  bias-center: "#9AA3B5"
  bias-leanright: "#F0A39B"
  bias-right: "#D9544A"
typography:
  display:
    fontFamily: '"Hanken Grotesk", system-ui, -apple-system, sans-serif'
    fontSize: "2.5rem"
    fontWeight: 800
    lineHeight: 1.1
    letterSpacing: "-0.02em"
  headline:
    fontFamily: '"Hanken Grotesk", system-ui, -apple-system, sans-serif'
    fontSize: "1.75rem"
    fontWeight: 800
    lineHeight: 1.2
    letterSpacing: "-0.01em"
  title:
    fontFamily: '"Hanken Grotesk", system-ui, -apple-system, sans-serif'
    fontSize: "1.25rem"
    fontWeight: 700
    lineHeight: 1.3
    letterSpacing: "normal"
  body:
    fontFamily: '"Hanken Grotesk", system-ui, -apple-system, sans-serif'
    fontSize: "0.875rem"
    fontWeight: 500
    lineHeight: 1.5
    letterSpacing: "normal"
  label:
    fontFamily: '"Hanken Grotesk", system-ui, -apple-system, sans-serif'
    fontSize: "0.6875rem"
    fontWeight: 700
    lineHeight: 1.2
    letterSpacing: "0.06em"
rounded:
  sm: "4px"
  md: "8px"
  lg: "12px"
  xl: "16px"
  full: "9999px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "16px"
  lg: "24px"
  xl: "32px"
components:
  button-primary:
    backgroundColor: "{colors.porcelain}"
    textColor: "{colors.midnight}"
    rounded: "{rounded.full}"
    padding: "8px 16px"
  input-claim:
    backgroundColor: "{colors.surface}"
    textColor: "{colors.porcelain}"
    rounded: "{rounded.xl}"
    padding: "16px"
---

# VeriFact AI Design System

## 1. Overview

VeriFact AI transforms ambiguous news claims, statements, and screenshots into evidence-backed credibility assessments in seconds. The interface embodies the **Midnight Transit Diagram** aesthetic: an authoritative cartographic wayfinding network where autonomous verification workers converge like transit lines into an interchange verdict hub.

- **Primary Goal:** Immediate clarity on claim credibility, uncertainty bounds, and primary sources within 3 seconds.
- **Tone:** Authoritative, clinical, transparent, and respectful of evidence limits.
- **Metaphor Rule:** Transit is the visual cartography, not the vocabulary. Critical user copy never employs playful transport puns.

## 2. Colors

### Palette Roles
- **Ground (Shell & Canvas):** Midnight Enamel `#0F1B3D`
- **Raised Surfaces (Cards & Rails):** `#16244D`
- **Type, Ticks & Core Accents:** Porcelain `#F4F1EA`

### Agent Line Cartography
- **Fact-Check Line:** Lilac `#B9A2F2` (top branch)
- **Cross-Reference Line:** Teal `#2BB3A3` (middle branch)
- **Linguistic Analysis Line:** Porcelain `#F4F1EA` with `#0F1B3D` navy casing (bottom branch)

### Verdict Bands & Uncertainty
- **Likely false (0–33):** Crimson `#DC2626`
- **Mixed (34–66):** Amber `#F59E0B`
- **Likely true (67–100):** Emerald `#10B981`
- **Not enough evidence (unverified):** Slate `#8A94B0` (Never displays composite digits)

### AllSides Media Bias Scale
- Left `#3B6FD8`, Lean Left `#8FB0F0`, Center `#9AA3B5`, Lean Right `#F0A39B`, Right `#D9544A`, Unrated neutral slate.

## 3. Typography

- **Primary Typeface:** Hanken Grotesk (`@fontsource/hanken-grotesk`), weights 500, 600, 700, 800.
- **Numbers & Metrics:** Tabular numerals enabled (`font-variant-numeric: tabular-nums`).
- **Hierarchy:**
  - Display: 40px / 800 weight on the interchange score.
  - Headline: 24–32px / 800 weight on claim quotes and section calls.
  - Body: 14–16px / 500–600 weight.
  - Station Labels: 11px uppercase bold tracking on cartographic lines.

## 4. Layout

- **Mobile First (375px+):** Single-column sequence: Claim Quote → Transit Interchange Diagram → Verdict Callout → Sources Rail → Language Radar → Google Search Suggestions → Verification Metadata.
- **Desktop (≥1024px):** Asymmetric 5-column grid:
  - Left Column (3/5): Claim, Transit Diagram, Verdict Line, and Weight Breakdown.
  - Right Column (2/5): Sticky Sources List, Language Radar, Search Suggestions, and Share Footer.
- **Maximum Width:** 1280px centered on viewports ≥1440px.

## 5. Elevation & Depth

VeriFact AI avoids skeuomorphic drop shadows, relying on tonal layering:
- Base layer: Midnight `#0F1B3D`
- Elevated cards: Surface `#16244D` with 1px border stroke `#24325C`
- Interactive overlays: Backdrop blur with 90% opacity

## 6. Shapes

- **Line Radii:** 8px stroke paths bending only at 0°, 45°, and 90°.
- **Interchange Circles:** Radius 58px fill with 4px outer verdict ring at radius 64px.
- **Station Nodes:** 6px radius circles with 3px stroke matching line ink.
- **Component Radii:** 12px for surface cards, 16px for input boxes, 9999px for pill chips and primary CTA buttons.

## 7. Components

- **TransitDiagram:** Signature SVG interchange visualizer. Handles checking animation (moving dash pattern) and result states (solid healthy lines vs dashed 50% opacity degraded agents).
- **ClaimInput:** 2000-character constrained textarea with real-time counter and rate-limiting lockout protection.
- **ScoreBreakdown:** Exact mathematical disclosure of $C = w_1 S_{fact} + w_2 (1 - B_{ling}) + w_3 V_{consensus}$ and confidence capping rules.
- **SourceList & SourceRow:** Segregated citations grouped strictly by finding agent (`found_by`), displaying publisher, AllSides bias rating, and stance chip.
- **LanguageRadar:** 4-axis sensory radar displaying Clickbait Syntax, Emotional Framing, Sensationalism, and Informational Density. Replaces chart with explicit notice when Linguistic analysis is unavailable.

## 8. Do's and Don'ts

### Do
- Always show "Not enough evidence" without any numerical score for `unverified` claims.
- Clearly indicate degraded worker agents with "Partial" status and dashed line rendering.
- Attribute AllSides media bias ratings with CC BY-NC 4.0 license link.
- Preserve typed claim text across validation and rate-limit errors.

### Don't
- Never invent or fabricate confidence scores when analysis agents fail.
- Never use transportation jargon ("delays", "rerouted", "station closed") in place of scientific claim outcomes.
- Never let LLMs guess media bias labels (AllSides static data only).
- Never render Google search suggestions without the required sandbox isolation.
