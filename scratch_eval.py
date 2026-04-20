import torch
from tasks.arc.arc_trainer import ARCModel
from tasks.arc.arc_loader import load_arc_dataset, grid_to_tensor_channels

device = torch.device('cpu')

def evaluate_arc(T, n):
    model = ARCModel(hidden_dim=512, T=T, n=n)
    model.load_state_dict(torch.load('checkpoints/arc/best_model.pt', map_location='cpu'))
    model.eval()
    
    tasks = load_arc_dataset('data/arc/data', split='training', max_tasks=1)
    task = tasks[0]
    
    MAX_H, MAX_W = 30, 30
    demo_input_list = []
    demo_output_list = []
    for pair in task["train"]:
        di = grid_to_tensor_channels(pair["input"], MAX_H, MAX_W)
        do = grid_to_tensor_channels(pair["output"], MAX_H, MAX_W)
        demo_input_list.append(torch.tensor(di))
        demo_output_list.append(torch.tensor(do))

    demo_inputs  = torch.stack(demo_input_list).unsqueeze(0).to(device)
    demo_outputs = torch.stack(demo_output_list).unsqueeze(0).to(device)
    demo_mask    = torch.ones(1, len(task["train"])).to(device)

    test_pair = task["test"][0]
    test_in = torch.tensor(grid_to_tensor_channels(test_pair["input"], MAX_H, MAX_W)).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, _ = model(demo_inputs, demo_outputs, demo_mask, test_in)
        
    pred = logits.argmax(dim=-1).squeeze(0).cpu().numpy()
    out_h, out_w = test_pair["output"].shape
    pred_cropped = pred[:out_h, :out_w]
    gt = test_pair["output"]
    
    match = (pred_cropped == gt).all()
    print(f"T={T}, n={n} | Match: {match}")
    print(f"Pred:\n{pred_cropped}")
    print(f"GT:\n{gt}")

print("Testing T=3, n=6")
evaluate_arc(3, 6)

print("\nTesting T=5, n=8")
evaluate_arc(5, 8)
