import torch
import torch.fx


class CustomTracerMobilenetV1(torch.fx.Tracer):
    def call_module(self, m, forward, args, kwargs):
        from onnx2torch.node_converters.pad import OnnxPadDynamic
        from onnx2torch.node_converters.reshape import OnnxReshape
        from onnx2torch.node_converters.constant_of_shape import OnnxConstantOfShape

        if isinstance(m, OnnxPadDynamic) and isinstance(args[1], torch.fx.Proxy):
            args = list(args)
            args[1] = torch.tensor([0, 0, 0, 0])
            if isinstance(args[2], torch.fx.Proxy):
                args[2] = 0.0
            args = tuple(args)

        if isinstance(m, OnnxReshape):
            args = list(args)
            args[1] = torch.tensor([1, 8])
            args = tuple(args)

        if isinstance(m, OnnxConstantOfShape):
            args = list(args)
            args[0] = torch.tensor([4])
            args = tuple(args)

        return super().call_module(m, forward, args, kwargs)

    def to_bool(self, v):
        return False

    def iter(self, obj: "Proxy"):
        return iter([0])


class CustomTracerGeneral(torch.fx.Tracer):
    def call_module(self, m, forward, args, kwargs):
        from onnx2torch.node_converters.pad import OnnxPadDynamic
        from onnx2torch.node_converters.base_element_wise import OnnxBaseElementWise
        from onnx2torch.node_converters.activations import OnnxSoftmaxV1V11
        from onnx2torch.node_converters.reshape import OnnxReshape
        from onnx2torch.node_converters.resize import OnnxResize
        from onnx2torch.node_converters.shape import OnnxShape

        if isinstance(m, OnnxPadDynamic) and isinstance(args[1], torch.fx.Proxy):
            args = list(args)
            args[1] = torch.tensor([0, 0, 0, 0])
            args = tuple(args)
        if isinstance(m, OnnxBaseElementWise):
            args = [torch.randn([1, 256, 56, 56]), torch.randn([1, 256, 56, 56])]
            args = tuple(args)
        if isinstance(m, OnnxSoftmaxV1V11):
            args = [torch.randn([1, 1000])]

        if isinstance(m, OnnxReshape):
            args = list(args)
            args[1] = torch.tensor([1, 2048])
            args = tuple(args)

        if isinstance(m, OnnxResize):
            args = list(args)
            args[0] = torch.randn([1, 1, 1, 2048])
            args = tuple(args)

        if isinstance(m, OnnxShape):
            args = list(args)
            args[0] = torch.randn([1, 1, 1, 2048])
            args = tuple(args)

        return super().call_module(m, forward, args, kwargs)

    def to_bool(self, v):
        return False

    def iter(self, obj: "Proxy"):
        return iter([0])


class CustomTracerYolo(torch.fx.Tracer):
    def call_module(self, m, forward, args, kwargs):
        from onnx2torch.node_converters.reshape import OnnxReshape
        from onnx2torch.node_converters.resize import OnnxResize
        from onnx2torch.node_converters.shape import OnnxShape

        if isinstance(m, OnnxReshape):
            args = list(args)
            if isinstance(args[1], torch.fx.Proxy):
                args[1] = torch.tensor([1])
            elif args[1].shape[0] == 5:
                if torch.all(torch.eq(args[1], torch.tensor([1, 1, 1, 2048, 1]))):
                    args[1] = torch.tensor([1])
            args = tuple(args)

        if isinstance(m, OnnxResize):
            args = list(args)
            args[0] = torch.randn([1, 1, 1, 2048])
            args = tuple(args)
            kwargs["scales"] = torch.tensor([1.0, 1.0, 2.0, 2.0])

        if isinstance(m, OnnxShape):
            args = list(args)
            args[0] = torch.randn([1, 1, 1, 2048])
            args = tuple(args)

        return super().call_module(m, forward, args, kwargs)

    def to_bool(self, v):
        return False
