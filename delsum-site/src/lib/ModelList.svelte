<script lang="ts">
    import DeleteButton from "./DeleteButton.svelte";
    import { removeValue, replaceValue } from "./list";
    import { models, stringToModelValue, type ModelDefinition } from "./model";
    import Model from "./Model.svelte";

    export type ModelItem = {
        id: number;
        model: ModelDefinition;
        value: Record<string, string>;
    };

    let {
        values,
        onUpdate = (_) => {},
        errors = {},
    }: {
        values: ModelItem[];
        onUpdate?: (input: ModelItem[]) => void;
        errors?: Record<number, string>;
    } = $props();

    let currentId = $derived((values[values.length - 1]?.id ?? -1) + 1);

    function addRow(model: string) {
        let newRow = {
            id: currentId,
            model: models[model],
            value: {},
        };
        onUpdate([...values, newRow]);
    }

    function addText(models: string) {
        let id = currentId;
        const lines = models.split("\n");
        let newModels: ModelItem[] = [];
        for (const line of lines) {
            try {
                const [model, value] = stringToModelValue(line);
                newModels.push({
                    id: id++,
                    model: model,
                    value: value,
                });
            } catch (error) {
                continue;
            }
        }
        onUpdate([...values, ...newModels]);
    }

    function processPaste(e: Event) {
        e.preventDefault();
        const event = e as ClipboardEvent;
        if (
            !event.clipboardData ||
            !event.clipboardData.types.includes("text/plain")
        ) {
            return;
        }
        const text = event.clipboardData.getData("text/plain");
        addText(text);
    }

    async function processDrop(event: DragEvent) {
        event.preventDefault();
        const files = event.dataTransfer?.files;
        if (!files) {
            return;
        }
        if (files && files.length > 0) {
            for (let i = 0; i < files.length; i++) {
                const file = files[i];
                const text = await file.text();
                addText(text);
            }
        }
    }

    function processDragover(event: DragEvent) {
        event.preventDefault();
    }
</script>

<div
    class="model-list"
    role="region"
    onpaste={processPaste}
    ondrop={processDrop}
    ondragover={processDragover}
>
    <div class="model-buttons">
        <button
            title="Add a new CRC model"
            onclick={() => {
                addRow("crc");
            }}>+CRC</button
        >
        <button
            title="Add a new modsum model"
            onclick={() => {
                addRow("modsum");
            }}>+modsum</button
        >
        <button
            title="Add a new fletcher model"
            onclick={() => {
                addRow("fletcher");
            }}>+fletcher</button
        >
        <button
            title="Add a new polyhash model"
            onclick={() => {
                addRow("polyhash");
            }}>+polyhash</button
        >

        <DeleteButton onDelete={() => onUpdate([])} />
    </div>
    <ul>
        {#each values as content (content.id)}
            <li>
                <Model
                    model={content.model}
                    value={content.value}
                    open={true}
                    onInput={(input) =>
                        onUpdate(
                            replaceValue({ ...content, value: input }, values),
                        )}
                    onDelete={() => onUpdate(removeValue(content.id, values))}
                    error={errors[content.id] ?? ""}
                />
            </li>
        {/each}
    </ul>
</div>

<style>
    .model-list {
        display: flex;
        flex-direction: column;
    }
    .model-buttons {
        display: flex;
        flex-direction: row;
    }
    .model-buttons button {
        flex: 1;
        border-radius: 0;
    }
</style>
