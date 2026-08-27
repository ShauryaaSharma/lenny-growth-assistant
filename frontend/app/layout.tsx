import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "The Lenny Growth Assistant",
  description:
    "Grounded answers, essays, and artifacts from Lenny's Podcast transcripts.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="h-full">
      <body className="h-full">
        {/* Keyboard users can jump past the session list straight to the chat. */}
        <a
          href="#composer"
          className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded focus:bg-accent focus:px-3 focus:py-2 focus:text-sm focus:text-white"
        >
          Skip to message input
        </a>
        {children}
      </body>
    </html>
  );
}
