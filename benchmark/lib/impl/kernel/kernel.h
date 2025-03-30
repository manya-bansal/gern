extern "C" {
void bli_sgemm_haswell_asm_6x16(
    signed long int m,
    signed long int n,
    signed long int k,
    const void *alpha,
    const void *a,
    const void *b,
    const void *beta,
    void *c,
    signed long int rs_c0,
    signed long int cs_c0,
    void *data,
    void *cntx);

void bli_sgemm_skx_asm_32x12_l2(
    signed long int m,
    signed long int n,
    signed long int k,
    const void *alpha,
    const void *a,
    const void *b,
    const void *beta,
    void *c,
    signed long int rs_c0,
    signed long int cs_c0,
    void *data,
    void *cntx);
}