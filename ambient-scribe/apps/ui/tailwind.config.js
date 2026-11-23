/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Modified by Super Protocol, 2026: the `@kui/foundations` plugin is replaced by
 * an explicit palette.
 *
 * Upstream's design tokens — `text-brand`, `bg-surface-raised`,
 * `bg-accent-blue-subtle` and about forty more — come from NVIDIA's Kaizen UI
 * foundations, vendored into this repository as `kui-foundations-0.403.0.tgz`
 * under the NVIDIA Kaizen UI License and loading the `nv-dark` theme. Neither
 * that tarball nor NVIDIA's brand palette may travel in an image we publish, so
 * the token NAMES are kept (the components reference them in hundreds of places)
 * and their VALUES are ours. Nothing about the layout changes; the colours are
 * a neutral slate-and-teal dark theme rather than NVIDIA's.
 */

/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: [
    './pages/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './app/**/*.{ts,tsx}',
    './src/**/*.{ts,tsx}',
  ],
  theme: {
    container: {
      center: true,
      padding: "2rem",
      screens: {
        "2xl": "1400px",
      },
    },
    extend: {
      colors: {
        // --- text ---------------------------------------------------------
        primary: "#e8edf2",
        secondary: "#9aa7b4",
        subtle: "#6b7885",
        inverse: "#0d1117",
        disabled: "#4a5561",

        // --- the product's own accent -------------------------------------
        brand: {
          DEFAULT: "#2dd4bf",
          hover: "#5eead4",
          subtle: "#113b38",
        },

        // --- surfaces -----------------------------------------------------
        surface: {
          base: "#0d1117",
          raised: "#161b22",
          sunken: "#090c10",
          subtle: "#1c232c",
        },

        // --- borders ------------------------------------------------------
        base: "#2a323c",

        // --- semantic accents, each with a `-subtle` fill and a `-hover` ---
        accent: {
          blue: { DEFAULT: "#58a6ff", subtle: "#102a43", hover: "#79b8ff" },
          green: { DEFAULT: "#3fb950", subtle: "#0f2913", hover: "#56d364" },
          red: { DEFAULT: "#f85149", subtle: "#2d1113", hover: "#ff7b72" },
          yellow: { DEFAULT: "#d29922", subtle: "#2b2109", hover: "#e3b341" },
          purple: { DEFAULT: "#bc8cff", subtle: "#231436", hover: "#d2a8ff" },
        },
      },
      borderRadius: {
        lg: "0.5rem",
        md: "0.375rem",
        sm: "0.25rem",
      },
      keyframes: {
        "accordion-down": {
          from: { height: 0 },
          to: { height: "var(--radix-accordion-content-height)" },
        },
        "accordion-up": {
          from: { height: "var(--radix-accordion-content-height)" },
          to: { height: 0 },
        },
      },
      animation: {
        "accordion-down": "accordion-down 0.2s ease-out",
        "accordion-up": "accordion-up 0.2s ease-out",
      },
    },
  },
  plugins: [
    require("tailwindcss-animate"),
    require("@tailwindcss/typography"),
  ],
}
