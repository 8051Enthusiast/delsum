import type { ModelItem } from "./ModelList.svelte";
import type { FileRowContent } from "./types";
import type { ChecksumRanges } from "./wasm/interfaces/delsum-web-checksums";

export interface ReverseRequestPayload {
    tag: "reverse";
    files: FileRowContent<string | File>[];
    models: ModelItem[];
    trailingCheck: boolean;
    extendedSearch: boolean;
}

export interface PartRequestPayload {
    tag: "part";
    files: FileRowContent<string | File>[];
    models: ModelItem[];
    trailingCheck: boolean;
    endRelative: boolean;
}

export interface CheckRequestPayload {
    tag: "checksum";
    files: FileRowContent<string | File>[];
    models: ModelItem[];
}

export type DelsumRequestPayload = PartRequestPayload | ReverseRequestPayload | CheckRequestPayload;

export interface DelsumMessage {
    id: number;
    tag: string;
}

export interface InputErrors {
    inputModelErrors: Record<number, string>;
    inputFileErrors: Record<number, string>;
}

export interface ReverseResponse extends DelsumMessage {
    tag: "reverse";
    models: string[];
    inputErrors: InputErrors;
}

export interface ChecksumRangeData {
    range: ChecksumRanges;
    model: string;
}

export interface PartResponse extends DelsumMessage {
    tag: "part";
    ranges: ChecksumRangeData[];
    inputErrors: InputErrors;
}

export interface Checks {
    checksums: string[][];
    fileLabels: string[];
    modelLabels: string[];
}

export interface CheckResponse extends DelsumMessage {
    tag: "checksum";
    checks: Checks;
    inputErrors: InputErrors;
}

export interface ErrorResponse extends DelsumMessage {
    tag: "error";
    error: string;
}

export interface DelsumRequest {
    id: number;
    payload: DelsumRequestPayload;
}

export type DelsumResponse = ReverseResponse | PartResponse | CheckResponse | ErrorResponse;
export type WorkerResponse = DelsumResponse;