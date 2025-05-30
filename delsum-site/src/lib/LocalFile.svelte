<script lang="ts">
    let {
        onInput: oninput = (_: File) => {},
        value,
    }: { onInput?: (_: File) => void; value: File } = $props();

    function handleInput(e: Event) {
        const element = e.target as HTMLInputElement;
        const file = element.files?.[0];
        if (file) {
            value = file;
            oninput(file);
        }
    }
</script>

<label>
    File
    <div class="button">{value.name}</div>
    <input type="file" oninput={handleInput} />
</label>

<style>
    input[type="file"] {
        display: none;
    }
    label .button {
        display: block;
        max-width: 20em;
        word-wrap: break-word;
        white-space: pre-wrap;
    }
</style>
