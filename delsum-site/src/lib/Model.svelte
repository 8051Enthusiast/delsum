<script lang="ts">
    import { modelValuesToString, type ModelDefinition } from "./model";
    import ModelRow from "./ModelRow.svelte";
    import Row from "./Row.svelte";

    let {
        model,
        value = {},
        onInput = (_) => {},
        onDelete = () => {},
        open = false,
        error = "",
    }: {
        model: ModelDefinition;
        value?: Record<string, string>;
        onInput?: (input: Record<string, string>) => void;
        onDelete?: () => void;
        open?: boolean;
        error?: string;
    } = $props();
</script>

<Row {onDelete} {error}>
    <details {open}>
        <summary
            >Model <div class="model-string">
                {modelValuesToString(model, value)}
            </div></summary
        >
        <div class="model-list">
            {#each model.rows as row (row.name)}
                <ModelRow
                    definition={row}
                    value={value[row.name] ?? ""}
                    onInput={(input) => {
                        onInput({ ...value, [row.name]: input });
                    }}
                />
            {/each}
        </div>
    </details>
</Row>
<style>
    .model-list {
        display: flex;
        flex-direction: row;
        flex-wrap: wrap;
    }
</style>
