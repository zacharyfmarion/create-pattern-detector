import { create } from "zustand";
import type { EntitySelection, LayerKey, Layers } from "./types";

export const DEFAULT_LAYERS: Layers = {
  gtGraph: true,
  predGraph: true,
  missingEdges: true,
  extraEdges: true,
  ambiguousEdges: true,
  weakEdges: true,
  shortEdges: true,
  crowdedVertices: true,
  evenDegree: true,
  kawasaki: true,
  maekawa: true,
  repairs: true,
  labels: false,
};

interface InspectorState {
  activeStage: string;
  selectedExampleKey: string | null;
  selectedWarningCode: string | null;
  selectedEntity: EntitySelection;
  layers: Layers;
  setActiveStage: (stage: string) => void;
  setSelectedExampleKey: (key: string | null) => void;
  setSelectedWarningCode: (code: string | null) => void;
  setSelectedEntity: (selection: EntitySelection) => void;
  toggleLayer: (key: LayerKey) => void;
}

export const useInspectorStore = create<InspectorState>((set) => ({
  activeStage: "stage5",
  selectedExampleKey: null,
  selectedWarningCode: null,
  selectedEntity: null,
  layers: DEFAULT_LAYERS,
  setActiveStage: (activeStage) =>
    set({ activeStage, selectedExampleKey: null, selectedEntity: null, selectedWarningCode: null }),
  setSelectedExampleKey: (selectedExampleKey) =>
    set({ selectedExampleKey, selectedEntity: null, selectedWarningCode: null }),
  setSelectedWarningCode: (selectedWarningCode) => set({ selectedWarningCode }),
  setSelectedEntity: (selectedEntity) => set({ selectedEntity }),
  toggleLayer: (key) =>
    set((state) => ({
      layers: { ...state.layers, [key]: !state.layers[key] },
    })),
}));
