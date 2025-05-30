<script lang="ts">
    import { FieldType, type ModelField } from "./model";

    let {
        definition,
        value = "",
        onInput = (_) => {},
    }: {
        definition: ModelField;
        value?: string;
        onInput?: (input: string) => void;
    } = $props();

    function processInput(e: Event) {
        const element = e.target as HTMLInputElement;
        const newValue = element.value;
        onInput(newValue);
    }
</script>

<div class="model-row">
    <label>
        {definition.name}
        {#if definition.type === FieldType.String}
            <input
                type="text"
                {value}
                class="model-input"
                oninput={processInput}
            />
        {:else if definition.type === FieldType.Decimal}
            <input
                type="text"
                {value}
                class="model-input"
                oninput={processInput}
                pattern="0|[1-9]+\d*"
            />
        {:else if definition.type === FieldType.Hexadecimal}
            <input
                type="text"
                {value}
                class="model-input"
                oninput={processInput}
                pattern="[0-9a-fA-F]*"
            />
        {:else if definition.type === FieldType.Boolean}
            <select {value} class="model-input" oninput={processInput}>
                <option value=""></option>
                <option value="true">true</option>
                <option value="false">false</option>
            </select>
        {:else if definition.type === FieldType.Endian}
            <select {value} class="model-input" oninput={processInput}>
                <option value=""></option>
                <option value="big">Big</option>
                <option value="little">Little</option>
            </select>
        {:else if definition.type === FieldType.Wordsize}
            <select {value} class="model-input" oninput={processInput}>
                <option value=""></option>
                <option value="8">8</option>
                <option value="16">16</option>
                <option value="24">24</option>
                <option value="32">32</option>
                <option value="40">40</option>
                <option value="48">48</option>
                <option value="56">56</option>
                <option value="64">64</option>
            </select>
        {:else if definition.type === FieldType.Signedness}
            <select {value} class="model-input" oninput={processInput}>
                <option value=""></option>
                <option value="signed">Signed</option>
                <option value="unsigned">Unsigned</option>
            </select>
        {/if}
    </label>
</div>

<style>
    .model-row {
        display: flex;
        flex-direction: row;
        text-align: right;
        min-width: 16em;
    }
    .model-input {
        width: 8em;
        padding: 2px;
        margin: 2px;
    }
    label {
        flex: 1;
    }
</style>
