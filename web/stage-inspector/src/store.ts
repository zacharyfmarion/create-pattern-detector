import { create } from "zustand";
import type { EntitySelection, LayerKey, Layers } from "./types";

export const DEFAULT_LAYERS: Layers = {
  gtGraph: false,
  predGraph: true,
  missingEdges: false,
  extraEdges: false,
  ambiguousEdges: false,
  weakEdges: false,
  shortEdges: false,
  crowdedVertices: false,
  evenDegree: false,
  kawasaki: false,
  maekawa: false,
  repairs: false,
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
  activeStage: "stage4",
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
