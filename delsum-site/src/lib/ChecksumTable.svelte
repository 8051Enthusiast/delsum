<script lang="ts">
    import type { Checks } from "./message";

    const { checksums }: { checksums: Checks } = $props();
</script>

<table>
    <thead>
        <tr>
            <th>File</th>
            {#each checksums.fileLabels as filename}
                <th>{filename}</th>
            {/each}
        </tr>
    </thead>
    <tbody>
        {#each checksums.modelLabels as model, i}
            <tr>
                <td><span class="model-string">{model}</span></td>
                {#each checksums.checksums[i] as checksum}
                    <td class="checksum">{checksum}</td>
                {/each}
            </tr>
        {/each}
    </tbody>
</table>

<style>
    table {
        width: 100%;
        border-collapse: collapse;
    }
    thead tr th {
	    background-color: var(--button);
	    border-color: var(--button);
	    color: var(--button-font);
        text-overflow: ellipsis;
    }
    tbody tr:nth-child(even) {
        background-color: var(--background-sec);
    }

    tbody tr:nth-child(odd) {
        background-color: var(--background);
    }
    .checksum {
        text-align: center;
        padding: 0.5em;
        font-family: monospace;
        font-size: large;
    }
</style>