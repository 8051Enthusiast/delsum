<script lang="ts">
  import "./lib/common.scss";
  import FileRowList from "./lib/FileRowList.svelte";
  import ModelList, { type ModelItem } from "./lib/ModelList.svelte";
  import { stringToModelValue } from "./lib/model";
  import type { FileRowContent } from "./lib/types";
  import type {
    Checks,
    ChecksumRangeData,
    DelsumRequest,
    DelsumRequestPayload,
    WorkerResponse,
  } from "./lib/message";
  import RangeList from "./lib/RangeList.svelte";
  import ChecksumTable from "./lib/ChecksumTable.svelte";
  type FileRow = FileRowContent<string | File>;
  let worker: Worker;
  type WorkerState = "ready" | "processing";
  let workerState: WorkerState = "ready";

  let messageId = 0;
  let files: FileRow[] = $state([
    {
      id: 0,
      file: "5813759a9972",
      checksum: "c4b28aa0",
    },
    {
      id: 1,
      file: "398798abca",
      checksum: "dfdf04ac",
    },
    {
      id: 2,
      file: "5979a979a797",
      checksum: "8aef53bb",
    },
  ]);
  const initialModel = stringToModelValue("crc width=32");
  let inputModels: ModelItem[] = $state([
    {
      id: 0,
      model: initialModel[0],
      value: initialModel[1],
    },
  ]);

  let trailingCheck: boolean = $state(false);
  let endRelative: boolean = $state(false);
  let extendedSearch: boolean = $state(false);

  let isCalculating: boolean = $state(true);
  let calculatingTimer: ReturnType<typeof setTimeout>;

  function updateState(newState: WorkerState) {
    if (newState === "ready") {
      clearTimeout(calculatingTimer);
      isCalculating = false;
    } else if (workerState === "ready") {
      clearTimeout(calculatingTimer);
      calculatingTimer = setTimeout(() => {
        isCalculating = true;
      }, 100);
    }
    workerState = newState;
  }

  type RequestTag = "part" | "reverse" | "checksum";
  let requestKind: RequestTag = $state("reverse");

  let resultKind: RequestTag | "error" | null = $state(null);
  let resultModels: string[] = $state([]);
  let resultRanges: ChecksumRangeData[] = $state([]);
  let resultChecksums: Checks = $state({
    checksums: [],
    fileLabels: [],
    modelLabels: [],
  });
  let resultError: string = $state("");
  let fileErrors: Record<number, string> = $state({});
  let modelErrors: Record<number, string> = $state({});

  function constructRequestPayload(): DelsumRequestPayload {
    let payload: DelsumRequestPayload;
    switch (requestKind) {
      case "part":
        payload = {
          tag: $state.snapshot(requestKind),
          files: $state.snapshot(files),
          models: $state.snapshot(inputModels),
          trailingCheck: $state.snapshot(trailingCheck),
          endRelative: $state.snapshot(endRelative),
        };
        break;
      case "reverse":
        payload = {
          tag: $state.snapshot(requestKind),
          files: $state.snapshot(files),
          models: $state.snapshot(inputModels),
          trailingCheck: $state.snapshot(trailingCheck),
          extendedSearch: $state.snapshot(extendedSearch),
        };
        break;
      case "checksum":
        payload = {
          tag: $state.snapshot(requestKind),
          files: $state.snapshot(files),
          models: $state.snapshot(inputModels),
        };
        break;
    }
    return payload;
  }

  function sendRequestUnchecked(
    request: DelsumRequestPayload,
    currentWorker: Worker = worker,
  ) {
    const message: DelsumRequest = {
      id: ++messageId,
      payload: request,
    };
    currentWorker.postMessage(message);
  }

  function createWorker() {
    let newWorker = new Worker(new URL("./lib/worker.ts", import.meta.url), {
      type: "module",
    });
    newWorker.onmessage = (e) => {
      const response: WorkerResponse = e.data;

      if (response.id != messageId) {
        return;
      }

      switch (response.tag) {
        case "part":
          resultRanges = response.ranges;
          resultKind = "part";
          fileErrors = response.inputErrors.inputFileErrors;
          modelErrors = response.inputErrors.inputModelErrors;
          break;
        case "reverse":
          resultModels = response.models;
          resultKind = "reverse";
          fileErrors = response.inputErrors.inputFileErrors;
          modelErrors = response.inputErrors.inputModelErrors;
          break;
        case "checksum":
          resultChecksums = response.checks;
          resultKind = "checksum";
          fileErrors = response.inputErrors.inputFileErrors;
          modelErrors = response.inputErrors.inputModelErrors;
          break;
        case "error":
          resultError = response.error;
          resultKind = "error";
      }
      updateState("ready");
    };
    return newWorker;
  }

  worker = createWorker();

  function sendRequest(request: DelsumRequestPayload) {
    switch (workerState) {
      case "processing":
        worker.terminate();
        worker = createWorker();
        break;
      case "ready":
        sendRequestUnchecked(request);
        updateState("processing");
        break;
    }
  }

  let requestTimer: ReturnType<typeof setTimeout> | null = null;

  function sendDebouncedRequest(request: DelsumRequestPayload) {
    // do it instantly the first time, otherwise do a timeout
    let time = 0;
    if (requestTimer !== null) {
      clearTimeout(requestTimer);
      time = 1000;
    }
    requestTimer = setTimeout(() => {
      sendRequest(request);
    }, time);
  }

  $effect(() => {
    const payload = constructRequestPayload();
    sendDebouncedRequest(payload);
  });
</script>

<main class="content">
  <h1>Delsum</h1>
  <p>
    Delsum is a reverse engineering tool for checksums.
  </p>
  <p>
    Here you can find out which checksum algorithm generate a given checksum, which regions of a file have a given checksum and more.
    Just input your files, and which checksum models you would like to consider.
  </p>
  <div class="inputs">
    <div class="box input-box">
      <h2>Files</h2>
      <FileRowList
        values={files}
        onUpdate={(newFiles) => (files = newFiles)}
        errors={fileErrors}
        hideChecksum={requestKind === "checksum" || trailingCheck}
      />
    </div>
    <div class="box input-box">
      <h2>Models</h2>
      <ModelList
        values={inputModels}
        onUpdate={(input: ModelItem[]) => (inputModels = input)}
        errors={modelErrors}
      />
    </div>
  </div>
  <div class="box">
    <h2>Options</h2>
    <div class="choosers">
      {#snippet kindChoice(kind: string, title: string)}
        <label
          class="edgy button {kind.toLowerCase() === requestKind
            ? 'chosen-one'
            : ''}"
          {title}
        >
          <input
            type="radio"
            name={kind.toLowerCase()}
            value={kind.toLowerCase()}
            bind:group={requestKind}
          />
          {kind}
        </label>
      {/snippet}
      {@render kindChoice(
        "Reverse",
        "Output all possible models where all files have matching checksums for any of the model prototypes given",
      )}
      {@render kindChoice(
        "Part",
        "For each model, give all possible ranges such that each range has the given checksum in every file",
      )}
      {@render kindChoice(
        "Checksum",
        "For each file and each model, returns the checksum of the whole file",
      )}
    </div>
    {#if requestKind !== "checksum"}
      <div>
        <label
          class="checkbox"
          title="Instead of explicitely giving a checksum, the last bytes of the input file are used as the input checksum"
        >
          <input type="checkbox" bind:checked={trailingCheck} />
          Input files end with checksum
        </label>
      </div>
    {/if}
    {#if requestKind === "part"}
      <div>
        <label
          class="checkbox"
          title="For cases where files have different sizes, instead of trying the first min(sizes) words for the range ends, try the last min(sizes) and output the offset as a negative number relative to the file end"
        >
          <input type="checkbox" bind:checked={endRelative} />
          Range ends are relative to file ends
        </label>
      </div>
    {/if}
    {#if requestKind === "reverse"}
      <div>
        <label
          class="checkbox"
          title="Try all possible parameter combinations, using more time and resulting in more false positives"
        >
          <input type="checkbox" bind:checked={extendedSearch} />
          Extended search
        </label>
      </div>
    {/if}
  </div>
  <div class="box">
    <h2>Output</h2>
    {#if !isCalculating}
      {#if resultKind === "reverse"}
        <ul class="output-list">
          {#each resultModels as model}
            <li class="output-item">
              <div class="model-string">{model}</div>
            </li>
          {/each}
        </ul>
        {#if resultModels.length === 0}
          <p>No matches found.</p>
        {/if}
      {:else if resultKind === "part"}
        <ul class="output-list">
          {#each resultRanges as range}
            <li class="output-item">
              <RangeList value={range} />
            </li>
          {/each}
        </ul>
        {#if resultRanges.length === 0}
          <p>No matches found.</p>
        {/if}
      {:else if resultKind === "checksum"}
        <ChecksumTable checksums={resultChecksums} />
      {:else if resultKind === "error"}
        <p class="error">{resultError}</p>
      {/if}
    {:else}
      <p>Calculating...</p>
    {/if}
  </div>
</main>

<style>
  .content {
    display: flex;
    flex-direction: column;
    flex-wrap: wrap;
  }
  .inputs {
    display: flex;
    flex-direction: row;
    flex-wrap: wrap;
    flex: 1;
  }
  .box {
    margin: 0.5em 1em;
    padding: 0.5em 1em 1em 1em;
    border-radius: 5px;
    background-color: var(--background-sec);
  }
  .input-box {
    flex: 1;
  }
  .output-list {
    display: flex;
    flex-direction: column;
    flex-wrap: wrap;
  }
  input[type="radio"] {
    display: none;
  }
  .chosen-one {
    background-color: var(--highlight);
    color: var(--foreground);
  }
  .choosers {
    display: flex;
    flex-direction: row;
    justify-content: center;
    flex: 1;
  }
</style>
