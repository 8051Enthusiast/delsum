<script lang="ts">
    let { onDelete = () => {} }: { onDelete: () => void } = $props();
</script>

<button
    class="close-button"
    onclick={() => {
        onDelete();
    }}
    title="Delete"
>✖</button>

<style>
    .close-button {
        border-radius: 0px;
        padding: 5px 6px;
        flex: 0;
        color: #888;
        border: none;
        transition:
            background-color 0.25s,
            color 0.25s;
        &:hover {
            background-color: #821;
            color: #fb9;
        }
    }
</style>
