<script lang="ts">
    import type { ChecksumRangeData } from "./message";
    import type { ChecksumRanges } from "./wasm/interfaces/delsum-web-checksums";
    const { value }: { value: ChecksumRangeData } = $props();
    let open = $state(false);

    function rangesToString(ranges: ChecksumRanges): string {
        const starts = ranges.start.join(",\u200b");
        const ends = ranges.end.join(",\u200b");
        return starts + ":\u200b" + ends;
    }
    function expandedRanges(ranges: ChecksumRanges): [number, number][] {
        let expanded: [number, number][] = [];
        let endIndex = 0;
        for (let i = 0; i < ranges.start.length; i++) {
            let startVal = ranges.start[i];
            while (
                endIndex < ranges.end.length &&
                ranges.end[endIndex] < startVal &&
                ranges.end[endIndex] >= 0
            ) {
                endIndex++;
            }
            if (endIndex >= ranges.end.length) {
                break;
            }

            for (let j = endIndex; j < ranges.end.length; j++) {
                let endVal = ranges.end[j];
                expanded.push([startVal, endVal]);
            }
        }
        return expanded;
    }

    const rangeText = $derived(rangesToString(value.range));
</script>

<details bind:open>
    <summary><span class="model-string">{value.model}</span>
        <div class="model-text">{rangeText}</div>
    </summary>
    {#if open}
        <ul>
            {#each expandedRanges(value.range) as [start, end]}
                <li>
                    <span>
                        {start}:{end}
                    </span>
                </li>
            {/each}
        </ul>
    {/if}
</details>

<style>
</style>
