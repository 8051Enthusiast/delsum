<script lang="ts">
    let {
        onInput: oninput = (_: string) => {},
        rowLength = 16,
        value = $bindable(""),
    }: {
        onInput?: (_: string) => void;
        rowLength?: number;
        value?: string;
    } = $props();

    function removeNonHex(text: string) {
        return text.replace(/[^0-9a-fA-F]/g, "");
    }

    function spaceHex(hexText: string) {
        return hexText.replace(/(.{2})/g, "$1 ").trimEnd();
    }

    function canonCursorPos(input: HTMLInputElement) {
        return input.selectionDirection === "backward"
            ? input.selectionStart
            : input.selectionEnd;
    }

    function paddedLength(length: number) {
        return (length * 3) >> 1;
    }

    const rowRegex = $derived(
        new RegExp(`.{1,${Math.floor(rowLength * 2)}}`, "g"),
    );
    const columns = $derived(paddedLength(Math.floor(rowLength * 2)));

    function formatHex(text: string) {
        const rows = text.match(rowRegex) ?? [text];
        return rows.map(spaceHex).join("\n");
    }

    const formattedHex = $derived(formatHex(value));

    function updateContent(text: string, cursorPos: number | null) {
        const pos = cursorPos ?? text.length;
        const front = removeNonHex(text.substring(0, pos));
        const back = removeNonHex(text.substring(pos, text.length));
        value = front + back;
        oninput(value);
        return paddedLength(front.length);
    }

    function handleInput(e: Event) {
        const element = e.target as HTMLInputElement;
        const cursorPos = canonCursorPos(element);
        const pos = updateContent(element.value, cursorPos);
        element.value = formattedHex;
        element.setSelectionRange(pos, pos);
    }
</script>

<label>
    <span> File Content </span>
    <textarea
        value={formattedHex}
        wrap="hard"
        cols={columns}
        rows={Math.min(Math.max(Math.ceil(formattedHex.length / columns), 1), 16)}
        class="hex-input"
        spellcheck="false"
        oninput={handleInput}
    ></textarea>
</label>

<style>
    .hex-input {
        resize: none;
    }
    label span {
        display: block;
    }
</style>
