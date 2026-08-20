// Guarded Firebase client init. This app runs and builds fine with zero
// NEXT_PUBLIC_FIREBASE_* env vars set (e.g. in CI, or before the user has
// created their own Firebase project) — in that case `auth` stays `null`
// and every auth-dependent feature must degrade to a "guest"/"not
// configured" state rather than throwing. Never hardcode real project
// config here.
import { initializeApp, getApps, type FirebaseApp } from "firebase/app";
import { getAuth, type Auth } from "firebase/auth";

const firebaseConfig = {
  apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY,
  authDomain: process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN,
  projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID,
  storageBucket: process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID,
  appId: process.env.NEXT_PUBLIC_FIREBASE_APP_ID,
};

const isConfigured = Boolean(firebaseConfig.apiKey && firebaseConfig.projectId);

let app: FirebaseApp | null = null;
let auth: Auth | null = null;

// Guard on `typeof window` too: the Firebase client SDK isn't meant to
// initialize during Next.js server-side rendering / static build.
if (isConfigured && typeof window !== "undefined") {
  app = getApps().length ? getApps()[0] : initializeApp(firebaseConfig);
  auth = getAuth(app);
}

export { app, auth, isConfigured as firebaseIsConfigured };
