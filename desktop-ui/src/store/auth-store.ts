import { Store } from "@tanstack/store";

const AUTH_STORAGE_KEY = "lune.desktop.auth";

export type AuthMode = "login" | "signup";

interface AuthProfile {
  email: string;
  name: string;
}

export interface AuthState {
  mode: AuthMode;
  profile: AuthProfile | null;
}

function loadStoredProfile(): AuthProfile | null {
  if (typeof window === "undefined") {
    return null;
  }

  const raw = window.localStorage.getItem(AUTH_STORAGE_KEY);

  if (!raw) {
    return null;
  }

  try {
    const parsed = JSON.parse(raw) as Partial<AuthProfile>;
    if (!parsed.email || !parsed.name) {
      return null;
    }
    return {
      email: parsed.email,
      name: parsed.name,
    };
  } catch {
    return null;
  }
}

function persistProfile(profile: AuthProfile | null) {
  if (typeof window === "undefined") {
    return;
  }

  if (!profile) {
    window.localStorage.removeItem(AUTH_STORAGE_KEY);
    return;
  }

  window.localStorage.setItem(AUTH_STORAGE_KEY, JSON.stringify(profile));
}

export function createInitialAuthState(): AuthState {
  return {
    mode: "login",
    profile: loadStoredProfile(),
  };
}

export const authStore = new Store(createInitialAuthState());

export const authActions = {
  setMode(mode: AuthMode) {
    authStore.setState((state) => ({ ...state, mode }));
  },

  signIn(email: string, name?: string) {
    const profile = {
      email,
      name: name?.trim() || email.split("@")[0] || "Lune user",
    };

    persistProfile(profile);
    authStore.setState((state) => ({
      ...state,
      profile,
    }));
  },

  signOut() {
    persistProfile(null);
    authStore.setState((state) => ({
      ...state,
      profile: null,
      mode: "login",
    }));
  },
};

export function isAuthenticated(state: AuthState) {
  return state.profile !== null;
}
