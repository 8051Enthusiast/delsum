import type { DelsumRequest, WorkerResponse, InputErrors, Checks, ChecksumRangeData } from "./message";
import { modelValuesToString } from "./model";
import type { ModelItem } from "./ModelList.svelte";
import type { FileRowContent } from "./types";
import { checksums, $init } from "./wasm/delsum_web";
import type { ChecksumError } from "./wasm/interfaces/delsum-web-checksums";
type FileRow = FileRowContent<string | File>;
function hexToBytes(hex: string): Uint8Array | null {
    if (hex.length % 2 != 0) {
        return null;
    }

    const bytes = new Uint8Array(hex.length / 2);
    for (let i = 0; i < hex.length; i += 2) {
        bytes[i / 2] = parseInt(hex.substring(i, i + 2), 16);
    }
    return bytes;
}

async function createChecksummedFile(
    file: FileRow,
): Promise<checksums.ChecksummedFile> {
    let fileContent: Uint8Array | null;
    let checksum: Uint8Array | null;

    if (typeof file.file === "string") {
        fileContent = hexToBytes(file.file);
        if (fileContent === null) {
            throw new Error("File hex must have an even number of digits");
        }
    } else {
        fileContent = new Uint8Array(await file.file.arrayBuffer());
    }

    checksum = hexToBytes(file.checksum);
    if (checksum === null) {
        throw new Error("Checksum must have an even number of digits");
    }

    return {
        file: fileContent,
        checksum: checksum,
    };
}

function reverseFilesForModel(
    checksummedFiles: checksums.ChecksummedFile[],
    model: ModelItem,
    trailingCheck: boolean,
    extendedSearch: boolean,
): string[] {
    const values = modelValuesToString(model.model, model.value);
    const models = checksums.reverse(
        checksummedFiles,
        values,
        trailingCheck,
        extendedSearch
    );
    return models;
}

interface ReverseResult {
    models: string[];
    inputErrors: InputErrors;
}

async function getFiles(f: FileRow[]) {
    let checksummedFiles: checksums.ChecksummedFile[] | null = [];
    let fileErrors: Record<number, string> = {};
    for (const row of f) {
        try {
            const result = await createChecksummedFile(row);
            if (checksummedFiles !== null) {
                checksummedFiles.push(result);
            }
        } catch (error) {
            const checksumError = error as Error;
            checksummedFiles = null;
            fileErrors[row.id] = checksumError.message;
        }
    }
    return { checksummedFiles, fileErrors };
}

async function reverseFiles(
    f: FileRow[],
    models: ModelItem[],
    trailingCheck: boolean,
    extendedSearch: boolean,
): Promise<ReverseResult> {
    const { checksummedFiles, fileErrors } = await getFiles(f);
    const modelErrors: Record<number, string> = {};
    if (checksummedFiles === null) {
        return {
            models: [],
            inputErrors: {
                inputModelErrors: [],
                inputFileErrors: fileErrors,
            },
        };
    }
    let res: string[] = [];
    for (const model of models) {
        try {
            const modelValues = reverseFilesForModel(checksummedFiles, model, trailingCheck, extendedSearch);
            res = res.concat(modelValues);
        } catch (error) {
            const checksumError = error as { payload: ChecksumError };
            switch (checksumError.payload.tag) {
                case "model":
                    modelErrors[model.id] = checksumError.payload.val;
                    break;
                case "other":
                    throw new Error(checksumError.payload.val);
            }
        }
    }
    return {
        models: res,
        inputErrors: {
            inputModelErrors: modelErrors,
            inputFileErrors: fileErrors,
        },
    };
}

function partFilesForModel(
    checksummedFiles: checksums.ChecksummedFile[],
    model: ModelItem,
    trailingCheck: boolean,
    endRelative: boolean,
): ChecksumRangeData[] {
    const values = modelValuesToString(model.model, model.value);
    const models = checksums.part(
        checksummedFiles,
        values,
        trailingCheck,
        endRelative
    );
    return models.map((model) => {
        return {
            range: model,
            model: values,
        };
    });
}

interface PartResult {
    ranges: ChecksumRangeData[];
    inputErrors: InputErrors;
}

async function partFiles(
    f: FileRow[],
    models: ModelItem[],
    trailingCheck: boolean,
    endRelative: boolean,
): Promise<PartResult> {
    const { checksummedFiles, fileErrors } = await getFiles(f);
    const modelErrors: Record<number, string> = {};
    let res: ChecksumRangeData[] = [];
    if (checksummedFiles === null) {
        return {
            ranges: [],
            inputErrors: {
                inputModelErrors: [],
                inputFileErrors: fileErrors,
            },
        };
    }
    for (const model of models) {
        try {
            const ranges = partFilesForModel(checksummedFiles, model, trailingCheck, endRelative);
            res = res.concat(ranges);
        } catch (error) {
            const { payload } = error as { payload: ChecksumError };
            switch (payload.tag) {
                case "model":
                    modelErrors[model.id] = payload.val;
                    break;
                case "other":
                    throw new Error(payload.val);
            }
        }
    }
    return {
        ranges: res,
        inputErrors: {
            inputModelErrors: modelErrors,
            inputFileErrors: fileErrors,
        },
    }
}

function checkFilesForModel(
    checksummedFiles: checksums.ChecksummedFile[],
    model: ModelItem,
): string[] {
    const modelString = modelValuesToString(model.model, model.value);
    let checksumResults: string[] = [];
    for (const file of checksummedFiles) {
        checksumResults.push(checksums.check(
            file.file,
            modelString,
        ));
    }
    return checksumResults;
}

interface CheckResult {
    checks: Checks;
    inputErrors: InputErrors;
}

function fileLabel(file: FileRow): string {
    if (typeof file.file === "string") {
        if (file.file.length > 10) {
            return file.file.substring(0, 10) + "...";
        }
        return file.file;
    } else {
        return file.file.name;
    }
}

async function checkFiles(
    f: FileRow[],
    models: ModelItem[],
): Promise<CheckResult> {
    const { checksummedFiles, fileErrors } = await getFiles(f);
    const fileLabels = f.map(fileLabel);
    const modelLabels = [];
    const modelErrors: Record<number, string> = {};
    let res: string[][] = [];
    if (checksummedFiles === null) {
        return {
            checks: {
                checksums: [],
                fileLabels: [],
                modelLabels: [],
            },
            inputErrors: {
                inputModelErrors: [],
                inputFileErrors: fileErrors,
            },
        };
    }
    for (const model of models) {
        try {
            const checks = checkFilesForModel(checksummedFiles, model);
            res.push(checks);
            if (!model.value["name"]) {
                modelLabels.push(modelValuesToString(model.model, model.value));
            } else {
                modelLabels.push(model.value["name"]);
            }
        } catch (error) {
            const checksumError = error as { payload: ChecksumError };
            switch (checksumError.payload.tag) {
                case "model":
                    modelErrors[model.id] = checksumError.payload.val;
                    break;
                case "other":
                    throw new Error(checksumError.payload.val);
            }
        }
    }
    return {
        checks: {
            checksums: res,
            fileLabels: fileLabels,
            modelLabels: modelLabels,
        },
        inputErrors: {
            inputModelErrors: modelErrors,
            inputFileErrors: fileErrors,
        },
    };
}

let isInit: boolean = false;
onmessage = async (e) => {
    const data: DelsumRequest = e.data;
    console.log("A");
    const payload = data.payload;
    if (!isInit) {
        // keep in mind that the other side makes sure
        // that only one request is in flight at the same time
        // (otherwise it just creates a new worker and terminates the old one)
        // otherwise this `isInit = true` could cause a concurrent second request
        // to continue without everything initialized
        // (and if the await was before the isInit, we would potentially do
        // await $init twice)
        isInit = true;
        await $init;
    }
    let response: WorkerResponse;
    try {
        switch (payload.tag) {
            case "part":
                const partResult = await partFiles(payload.files, payload.models, payload.trailingCheck, payload.endRelative);
                response = {
                    id: data.id,
                    tag: "part",
                    ...partResult,
                }
                break;
            case "reverse":
                const reverseResult = await reverseFiles(payload.files, payload.models, payload.trailingCheck, payload.extendedSearch);
                response = {
                    id: data.id,
                    tag: "reverse",
                    ...reverseResult,
                };
                break;
            case "checksum":
                const checkResult = await checkFiles(payload.files, payload.models);
                response = {
                    id: data.id,
                    tag: "checksum",
                    ...checkResult,
                };
                break;
        }
    } catch (error) {
        response = {
            id: data.id,
            tag: "error",
            error: String(error),
        };
    }
    console.log(isInit);
    postMessage(response);
}