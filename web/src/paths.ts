const BASE_PATH = import.meta.env.BASE_URL.endsWith("/")
  ? import.meta.env.BASE_URL
  : `${import.meta.env.BASE_URL}/`;

export const STATIC_REPLAYS = import.meta.env.VITE_STATIC_REPLAYS === "1";

export function appPath(path: string): string {
  return `${BASE_PATH}${path.replace(/^\//, "")}`;
}

export function currentRoutePath(): string {
  const base = new URL(BASE_PATH, window.location.origin).pathname;
  const pathname = window.location.pathname;
  let route = pathname;
  if (base !== "/" && pathname.startsWith(base)) {
    route = `/${pathname.slice(base.length)}`;
  }
  return route.length > 1 ? route.replace(/\/+$/, "") : route;
}
