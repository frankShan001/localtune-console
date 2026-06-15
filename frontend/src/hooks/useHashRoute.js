import { useEffect, useState } from "react";

function parseHashRoute() {
  return (window.location.hash.replace("#/", "").split("?")[0] || "overview");
}

export default function useHashRoute() {
  const [route, setRoute] = useState(parseHashRoute);
  useEffect(() => {
    const onHashChange = () => setRoute(parseHashRoute());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);
  return [route, (next) => {
    window.location.hash = `#/${next}`;
    setRoute(next);
  }];
}
