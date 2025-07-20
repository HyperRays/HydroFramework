
// Minimal compute shader for buffer add
[[vk::binding(0, 0)]] 
RWStructuredBuffer<uint32_t> data;

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    data[id.x] = data[id.x] + 1;
}
