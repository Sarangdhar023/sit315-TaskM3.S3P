
__kernel void vector_add_ocl(const int size, __global int *v1, __global int *v2, __global int *v_out) {

    
    const int globalIndex = get_global_id(0);

  
    if (globalIndex < size) {

        int index = clamp(globalIndex, 0, size - 1);
        v_out[index] = v1[index] + v2[index];
    }
}
