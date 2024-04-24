"use client";

import { Inter } from "next/font/google";
import "./globals.css";
import "@radix-ui/themes/styles.css";
import { Theme, Button } from "@radix-ui/themes";
import { GitHubLogoIcon } from "@radix-ui/react-icons";

import { ReactNode } from "react";
import { SiteMeta } from "@/src/components/SiteMeta/SIteMeta";

const inter = Inter({ subsets: ["latin"] });

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning className="dark">
      <head>
        <SiteMeta />
      </head>
      <body>
        <Theme hasBackground={false} appearance="dark" accentColor="iris">
          <div>{children}</div>
          <a
            href="https://github.com/DaveBitter/tensorflow-transfer-learning"
            target="__blank"
            className="fixed top-8 right-8"
          >
            <GitHubLogoIcon width="32" height="32" />
          </a>
        </Theme>
      </body>
    </html>
  );
}
