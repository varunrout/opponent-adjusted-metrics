"use client";

import { useState, type FormEvent } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { GoogleAuthProvider, signInWithEmailAndPassword, signInWithPopup } from "firebase/auth";
import { PitchMap } from "@/components/ui/PitchMap";
import { auth } from "@/lib/firebase";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const authConfigured = Boolean(auth);

  async function handleGoogleSignIn() {
    if (!auth) return;
    setError(null);
    try {
      await signInWithPopup(auth, new GoogleAuthProvider());
      router.push("/overview");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Google sign-in failed.");
    }
  }

  async function handleEmailSignIn(e: FormEvent) {
    e.preventDefault();
    if (!auth) return;
    setError(null);
    setSubmitting(true);
    try {
      await signInWithEmailAndPassword(auth, email, password);
      router.push("/overview");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Sign-in failed. Check your email and password.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="relative min-h-screen flex items-center justify-center overflow-hidden" style={{ background: "var(--bg)" }}>
      <div className="absolute inset-0 opacity-20 pointer-events-none">
        <PitchMap />
      </div>

      <div
        className="relative z-10 w-full max-w-sm bg-card border border-border rounded p-6"
        style={{ color: "var(--text)" }}
      >
        <h1 className="text-[15px] font-semibold m-0 mb-1">Sign in</h1>
        <p className="text-[12.5px] text-text2 m-0 mb-5">Opponent-Adjusted Metrics</p>

        {!authConfigured && (
          <p className="text-[12px] mb-4" style={{ color: "var(--amber)" }}>
            Sign-in isn&apos;t configured in this environment yet.
          </p>
        )}

        <button
          type="button"
          onClick={handleGoogleSignIn}
          disabled={!authConfigured}
          className="w-full rounded-lg px-3 py-2 text-[13px] font-medium bg-teal disabled:opacity-40 disabled:cursor-not-allowed"
          style={{ color: "#04231f" }}
        >
          Continue with Google
        </button>

        <div className="flex items-center gap-2 my-4">
          <div className="h-px flex-1" style={{ background: "var(--border)" }} />
          <span className="text-[11px] text-text2">or</span>
          <div className="h-px flex-1" style={{ background: "var(--border)" }} />
        </div>

        <form onSubmit={handleEmailSignIn} className="flex flex-col gap-2">
          <input
            type="email"
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            disabled={!authConfigured}
            className="bg-surface text-text border border-border rounded-lg px-2.5 py-1.5 text-[13px] disabled:opacity-40"
            aria-label="Email"
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            disabled={!authConfigured}
            className="bg-surface text-text border border-border rounded-lg px-2.5 py-1.5 text-[13px] disabled:opacity-40"
            aria-label="Password"
          />
          <button
            type="submit"
            disabled={!authConfigured || submitting}
            className="rounded-lg px-3 py-1.5 text-[12.5px] bg-transparent border border-border text-text2 disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {submitting ? "Signing in…" : "Sign in"}
          </button>
        </form>

        {error && (
          <p className="text-[12px] mt-3 mb-0" style={{ color: "var(--red)" }}>
            {error}
          </p>
        )}

        <div className="mt-5 pt-4 text-center" style={{ borderTop: "1px solid var(--border)" }}>
          <Link href="/overview" className="text-[12px] text-text2 underline">
            Continue as guest
          </Link>
        </div>
      </div>
    </div>
  );
}
