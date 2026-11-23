/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Super Protocol
 * SPDX-License-Identifier: Apache-2.0
 *
 * This file is ours. It carries no upstream copyright line because none of it is
 * upstream's code: it is the four `@kui/react` components this application used,
 * rebuilt from their call sites.
 *
 * `@kui/react` is NVIDIA's Kaizen UI, vendored into this repository as a
 * tarball under the NVIDIA Kaizen UI License — "the distributable portions of
 * the Software shall only be accessed by your application", which a published
 * container image is not a safe reading of. It is also, in this application,
 * four components: a theme wrapper, a header bar, a button and a text span. They
 * are here, with the same props the call sites pass, so `App.tsx` changes by one
 * import line.
 */
import type { ReactNode } from 'react'

export function ThemeProvider({ children }: { children: ReactNode }) {
  return <div className="contents">{children}</div>
}

export function AppBar({ slotLeft, slotRight }: { slotLeft?: ReactNode; slotRight?: ReactNode }) {
  return (
    <header className="flex items-center justify-between gap-4 border-b border-base bg-surface-raised px-4 py-2">
      <div className="flex items-center">{slotLeft}</div>
      <div className="flex items-center">{slotRight}</div>
    </header>
  )
}

type TextKind =
  | 'title/md' | 'title/sm' | 'label/regular/xs' | 'body/regular/sm' | string

const TEXT_KINDS: Record<string, string> = {
  'title/md': 'text-lg font-semibold',
  'title/sm': 'text-base font-semibold',
  'label/regular/xs': 'text-xs',
  'body/regular/sm': 'text-sm',
}

export function Text({
  kind, className = '', children, ...rest
}: { kind?: TextKind; className?: string; children?: ReactNode } & React.HTMLAttributes<HTMLSpanElement>) {
  const base = (kind && TEXT_KINDS[kind]) || 'text-sm'
  return <span className={`${base} ${className}`.trim()} {...rest}>{children}</span>
}

const BUTTON_SIZES: Record<string, string> = {
  small: 'px-2.5 py-1 text-xs',
  medium: 'px-3 py-1.5 text-sm',
  large: 'px-4 py-2.5 text-sm',
}

export function Button({
  kind = 'primary', color = 'brand', size = 'medium', className = '',
  children, ...rest
}: {
  kind?: 'primary' | 'secondary' | 'tertiary'
  color?: 'brand' | 'neutral'
  size?: 'small' | 'medium' | 'large'
  className?: string
  children?: ReactNode
} & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  const look =
    kind === 'primary'
      ? (color === 'brand'
          ? 'bg-brand text-inverse hover:bg-brand/90'
          : 'bg-surface-subtle text-primary hover:bg-surface-raised')
      : kind === 'secondary'
        ? 'border border-base text-primary hover:bg-surface-subtle'
        : 'text-brand hover:text-brand-hover'
  return (
    <button
      className={[
        'inline-flex items-center justify-center gap-2 rounded-md font-medium',
        'transition-colors focus:outline-none focus:ring-2 focus:ring-brand/50',
        'disabled:cursor-not-allowed disabled:opacity-50',
        BUTTON_SIZES[size] || BUTTON_SIZES.medium, look, className,
      ].join(' ').trim()}
      {...rest}
    >
      {children}
    </button>
  )
}
