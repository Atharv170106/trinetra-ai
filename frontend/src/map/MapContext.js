import { createContext, useContext } from "react";

export const MapContext = createContext(null);

export function useMap() {
  const context = useContext(MapContext);
  if (context === undefined) {
    throw new Error("useMap must be used within a MapContext.Provider (like <MapContainer>)");
  }
  return context;
}
