import type { Metadata } from "next";
import { Montserrat, Inconsolata } from "next/font/google";
import "./globals.css";
import { RoleProvider } from "@/components/shell/RoleProvider";
import { RoleGate } from "@/components/shell/RoleGate";
import { TopBar } from "@/components/shell/TopBar";
import { Sidebar } from "@/components/shell/Sidebar";
import { MatchFilterProvider } from "@/components/shell/MatchFilterProvider";

const montserrat = Montserrat({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-montserrat",
  display: "swap",
});

const inconsolata = Inconsolata({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-inconsolata",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Opponent-Adjusted Metrics",
  description: "Football analytics, adjusted for who you're playing.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${montserrat.variable} ${inconsolata.variable}`}>
      <body className="font-ui">
        <RoleProvider defaultRole="guest">
          <MatchFilterProvider>
            <RoleGate />
            <TopBar />
            <div className="flex" style={{ minHeight: "calc(100vh - 56px)" }}>
              <Sidebar />
              <main className="flex-1 px-7 py-6 overflow-hidden">{children}</main>
            </div>
          </MatchFilterProvider>
        </RoleProvider>
      </body>
    </html>
  );
}
