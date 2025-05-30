<script lang="ts" generics="T">
    import type { Component } from "svelte";
    import type { FileRowContent } from "./types.js";
    import Row from "./Row.svelte";

    type OnInputHandler = (input: FileRowContent<T>) => void;
    type OnFileInputHandler = (input: T) => void;
    type OnDeleteHandler = (id: number) => void;
    type Props = {
        onInput: OnFileInputHandler;
        value: T;
    };

    let {
        File,
        value,
        error = "",
        hideChecksum = false,
        onInput = (_) => {},
        onDelete = (_) => {},
        ...others
    }: {
        File: Component<Props>;
        value: FileRowContent<T>;
        error?: string;
        hideChecksum?: boolean;
        onInput?: OnInputHandler;
        onDelete?: OnDeleteHandler;
    } & Record<string, unknown> = $props();

    function onFileInput(input: T) {
        onInput({ ...value, file: input });
    }

    function onChecksumInput(event: Event) {
        const target = event.target as HTMLInputElement;
        const newValue = target.value;
        onInput({ ...value, checksum: newValue });
    }
</script>

<Row {error} onDelete={() => onDelete(value.id)}>
    <div class="content-input">
        <File onInput={onFileInput} value={value.file} {...others} />
    </div>
    {#if !hideChecksum}
        <div class="checksum-input">
            <label>
                <span> Checksum </span>
                <input
                    type="text"
                    size={16}
                    value={value.checksum}
                    oninput={onChecksumInput}
                    class="checksum-text"
                />
            </label>
        </div>
    {/if}
</Row>

<style>
    .content-input {
        display: flex;
        flex: 1;
        float: left;
        padding: 5px;
        text-align: left;
        min-height: 4em;
    }
    .checksum-input {
        padding: 5px;
    }
    label span {
        display: block;
    }
</style>
