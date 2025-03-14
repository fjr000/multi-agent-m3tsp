# This is a sample Python script.
import numpy as np


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    def generate_agent_masks(original_mask):
        num_agents = len(original_mask)
        new_masks = []

        for i in range(num_agents):
            if not original_mask[i]:
                # False的智能体：原mask取反
                new_mask = [not val for val in original_mask]
            else:
                # True的智能体：仅自己为True
                new_mask = [False] * num_agents
                new_mask[i] = True
            new_masks.append(new_mask)
        return new_masks

    def get_masks_in_salesmen(original_mask):
        B, A = original_mask.shape
        global_invert = ~original_mask  # Shape [B, A]
        result = np.zeros((B, A, A), dtype=bool)

        # 处理False的情况
        # 找到所有批次和代理中为False的位置
        false_batch, false_agents = np.where(global_invert)
        # 对于每个这样的位置，设置对应的行为global_invert的对应批次
        result[false_batch, false_agents, :] = global_invert[false_batch, :]

        # 处理True的情况
        true_batch, true_agents = np.where(original_mask)
        result[true_batch, true_agents, true_agents] = True

        return result


    # 验证示例
    original = np.array([False, True, False, False, True, True, False])
    result = generate_agent_masks(original)
    results = get_masks_in_salesmen(original[None,:])

    # 格式化输出
    print("[")
    for mask in result:
        print(f" {str(mask).replace('False', 'false').replace('True', 'true')},")
    print("]")

    # 格式化输出
    print("[")
    for mask in results[0]:
        print(f" {str(mask).replace('False', 'false').replace('True', 'true')},")
    print("]")


    for
