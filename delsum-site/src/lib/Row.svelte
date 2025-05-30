<script lang="ts">
    import type { Snippet } from "svelte";
    import DeleteButton from "./DeleteButton.svelte";

    let {
        onDelete = () => {},
        children,
        error = "",
    }: { onDelete: () => void; children: Snippet; error?: string } = $props();
</script>

<div class="columns">
    <div class="rows full-flex">
        <div class="columns full-flex">
            {@render children()}
        </div>
        <span class="error">{error}</span>
    </div>
    <DeleteButton {onDelete} />
</div>

<style>
    .columns {
        display: flex;
        flex-direction: row;
        justify-content: space-between;
    }
    .rows {
        display: flex;
        flex-direction: column;
    }
    .full-flex {
        flex: 1
    }
</style>
