"""Export small fixed reference cases; this does not calibrate or train hardware."""
import argparse
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from race_rl.serialization import save_json,code_hashes
from race_rl.accelerators.cnn_ppa import cnn_hardware_record
from race_rl.accelerators.gemmini_model import GemminiAction,execute_gemm
from race_rl.accelerators.cnn_quantization import calibrate,execute_quantized,tiled_integer_conv,integer_conv_reference
from race_rl.accelerators.cnn_design import LayerPlan
from race_rl.frontend import from_torch,array_hash


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=Path('experiments/runs/cnn_calibration_cases'))
    args=parser.parse_args(argv)
    if args.out.exists() and any(args.out.iterdir()):parser.error('choose a fresh --out')
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);torch.manual_seed(23)
    rng=np.random.default_rng(23);cases=[]
    for name,shape in (('gemm_16',(16,16,16)),('gemm_tail',(29,17,33)),('gemm_reuse',(64,32,64))):
        m,n,k=shape;a=rng.integers(-4,5,(m,k),dtype=np.int8);b=rng.integers(-4,5,(k,n),dtype=np.int8)
        result=execute_gemm(a,b,GemminiAction())
        expected=a.astype(np.int64)@b.astype(np.int64)
        if not np.array_equal(result,expected):raise ValueError('independent GEMM mismatch')
        path=args.out/(name+'.npz');np.savez_compressed(path,A=a,B=b,expected=result)
        cases.append({'case':name,'kind':'INT8 GEMM / INT32','file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                      'result_sha256':array_hash(result),'shape':shape,'numerical_status':'PASSED',
                      'observed_cycles':None,'measured_energy_j':None,'rtl_status':'NOT_RUN'})
    for name,groups,stride,dilation in (('conv_grouped',2,1,2),('conv_depthwise',4,2,1)):
        model=torch.nn.Sequential(torch.nn.Conv2d(4,4,3,stride=stride,padding=dilation,dilation=dilation,groups=groups,bias=False)).eval()
        x=rng.integers(-4,5,(1,4,9,9),dtype=np.int8);w=rng.integers(-4,5,(4,4//groups,3,3),dtype=np.int8)
        imported=from_torch(model,torch.from_numpy(x.astype(np.float32)))
        op=imported.workload.operators[0]
        tiled=tiled_integer_conv(imported.workload,op,x,w,LayerPlan())
        expected=integer_conv_reference(x,w,op)
        if not np.array_equal(tiled,expected):raise ValueError('HWC/K16 reconstruction mismatch')
        path=args.out/(name+'.npz');np.savez_compressed(path,input=x,weights=w,expected=expected)
        cases.append({'case':name,'kind':'grouped/dilated HWC gather + K16 reduction','file':path.name,
                      'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'operator':asdict(op),
                      'result_sha256':array_hash(expected),'numerical_status':'PASSED','observed_cycles':None,'rtl_status':'NOT_RUN'})
    class VectorFixture(torch.nn.Module):
        def __init__(self):
            super().__init__();self.bn=torch.nn.BatchNorm2d(3);self.relu=torch.nn.ReLU6()
            self.avg=torch.nn.AvgPool2d(2);self.max=torch.nn.MaxPool2d(2);self.adapt=torch.nn.AdaptiveAvgPool2d(1)
        def forward(self,x):return self.adapt(self.avg(self.relu(self.bn(x)))+self.max(x))
    example=torch.randn(1,3,8,8);bundle=calibrate(from_torch(VectorFixture().eval(),example),example)
    output,numerical=execute_quantized(bundle,example.numpy())
    path=args.out/'vector_postops.npz';np.savez_compressed(path,input=example.numpy(),expected=output,**bundle.parameters)
    save_json(args.out/'vector_quantization.json',bundle.manifest)
    cases.append({'case':'vector_postops','kind':'FP32 BN/ReLU6/pooling/Add/requantize','file':path.name,
                  'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'numerical_reference':numerical,
                  'observed_cycles':None,'measured_energy_j':None,'rtl_status':'PROJECT_EXTENSION_NOT_IMPLEMENTED'})
    record={'status':'REFERENCE_CASES_EXPORTED_NOT_CALIBRATED','hardware':cnn_hardware_record(),
            'source_hashes':code_hashes(),'seed':23,'cases':cases,
            'required_measurements':{'process_node':None,'clock_hz':None,'voltage_v':None,
                'complete_chiplet_area_mm2':None,'gb_lb_rf_acc_access_costs_j':None,'vector_op_j':None,
                'mac_j':None,'dma_and_mesh_costs_j':None,'static_power_w':None},
            'acceptance':'bind exact hardware identity; independently verify numeric outputs, capacity and port traces; compare actual RTL cycles case by case; report error distributions; obtain technology-specific energy/area sources',
            'limits':'Gemmini upstream RTL lacks this independent LB and vector extension. Running unchanged Gemmini RTL cannot calibrate the whole custom CNN architecture. This export does not enable formal PPA or accuracy claims.'}
    save_json(args.out/'manifest.json',record)
    print('已輸出 6 個小型校準參照案例；沒有執行 RTL、量測 PPA 或訓練。')
    print(args.out/'manifest.json')
    return 0


if __name__=='__main__':raise SystemExit(main())
