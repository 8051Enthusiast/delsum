<script lang="ts">
    import FileRow from "./FileRow.svelte";
    import Hex from "./Hex.svelte";
    import LocalFile from "./LocalFile.svelte";
    import DeleteButton from "./DeleteButton.svelte";
    import type { FileRowContent } from "./types";
    import { containsFile, containsString } from "./types";
    import { removeValue, replaceValue } from "./list";
    type Row = FileRowContent<string | File>;
    type UpdateHandler = (values: Row[]) => void;

    let {
        values,
        onUpdate = (_) => {},
        errors = {},
        hideChecksum = false,
    }: {
        values: Row[];
        onUpdate?: UpdateHandler;
        errors?: Record<number, string>;
        hideChecksum?: boolean;
    } = $props();
    let currentId = $derived((values[values.length - 1]?.id ?? -1) + 1);

    function addFiles(files: FileList, values: Row[]) {
        let id = currentId;
        const newFiles = Array.from(files).map((file) => ({
            id: id++,
            file: file,
            checksum: "",
        }));
        return [...values, ...newFiles];
    }

    function processDrop(event: DragEvent) {
        event.preventDefault();
        const files = event.dataTransfer?.files;
        if (files) {
            onUpdate(addFiles(files, values));
        }
    }

    function processDragover(event: DragEvent) {
        event.preventDefault();
    }
</script>

<div
    class="file-row-list"
    role="region"
    ondrop={processDrop}
    ondragover={processDragover}
>
    <div class="file-row-buttons">
        <button
            title="Add a hex field for inputting bytes manually"
            onclick={() => {
                let newRow = { id: currentId, checksum: "", file: "" };
                onUpdate([...values, newRow]);
            }}>+Hex</button
        >

        <label
            class="button"
            title="Add some files from disk"
            >+Files<input
                multiple
                type="file"
                class="button"
                oninput={(e: Event) => {
                    const element = e.target as HTMLInputElement;
                    if (element.files) {
                        onUpdate(addFiles(element.files, values));
                    }
                }}
            />
        </label>
        <DeleteButton onDelete={() => onUpdate([])} />
    </div>
    <ul>
        {#each values as content (content.id)}
            <li>
                {#if containsString(content)}
                    <FileRow
                        File={Hex}
                        value={content}
                        error={errors[content.id] ?? ""}
                        {hideChecksum}
                        onInput={(newInput) => {
                            onUpdate(replaceValue(newInput, values));
                        }}
                        onDelete={(deleteId) => {
                            onUpdate(removeValue(deleteId, values));
                        }}
                        rowCount={16}
                    />
                {:else if containsFile(content)}
                    <FileRow
                        File={LocalFile}
                        value={content}
                        error={errors[content.id] ?? ""}
                        {hideChecksum}
                        onInput={(newInput) => {
                            onUpdate(replaceValue(newInput, values));
                        }}
                        onDelete={(deleteId) => {
                            onUpdate(removeValue(deleteId, values));
                        }}
                    />
                {/if}
            </li>
        {/each}
    </ul>
</div>

<style>
    .file-row-list {
        display: flex;
        flex-direction: column;
    }
    .file-row-buttons {
        display: flex;
        flex-direction: row;
        min-width: 32em;
    }
    input[type="file"] {
        display: none;
    }
    .file-row-buttons button,
    .file-row-buttons label {
        flex: 1;
        border-radius: 0;
    }
</style>
